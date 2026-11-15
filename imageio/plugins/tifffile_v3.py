"""Read/Write TIFF files using tifffile.

.. note::
    To use this plugin you need to have `tifffile
    <https://github.com/cgohlke/tifffile>`_ installed::

        pip install tifffile

This plugin wraps tifffile, a powerful library to manipulate TIFF files. It
superseeds our previous tifffile plugin and aims to expose all the features of
tifffile.

The plugin treats individual TIFF series as ndimages. A series is a sequence of
TIFF pages that, when combined describe a meaningful unit, e.g., a volumetric
image (where each slice is stored on an individual page) or a multi-color
staining picture (where each stain is stored on an individual page). Different
TIFF flavors/variants use series in different ways and, as such, the resulting
reading behavior may vary depending on the program used while creating a
particular TIFF file.

Methods
-------
.. note::
    Check the respective function for a list of supported kwargs and detailed
    documentation.

.. autosummary::
    :toctree:

    TifffilePlugin.read
    TifffilePlugin.iter
    TifffilePlugin.write
    TifffilePlugin.properties
    TifffilePlugin.metadata

Additional methods available inside the :func:`imopen <imageio.v3.imopen>`
context:

.. autosummary::
    :toctree:

    TifffilePlugin.iter_pages

"""

from io import BytesIO
from typing import Any, Dict, Iterator, Optional, Tuple, cast
import inspect
import math
import warnings

import numpy as np
import tifffile

from ..core.request import URI_BYTES, InitializationError, Request
from ..core.v3_plugin_api import ImageProperties, PluginV3
from ..typing import ArrayLike


def _get_resolution(page: tifffile.TiffPage) -> Dict[str, Any]:
    metadata = {}

    try:
        metadata["resolution_unit"] = page.tags[296].value.value
    except KeyError:
        # tag 296 missing
        return metadata

    try:
        resolution_x = page.tags[282].value
        resolution_y = page.tags[283].value

        metadata["resolution"] = (
            resolution_x[0] / resolution_x[1],
            resolution_y[0] / resolution_y[1],
        )
    except KeyError:
        # tag 282 or 283 missing
        pass
    except ZeroDivisionError:
        warnings.warn(
            "Ignoring resolution metadata because at least one direction has a 0 "
            "denominator.",
            RuntimeWarning,
        )

    return metadata


def _normalize_region(
    region: Tuple[int, int, int, int], imagelength: int, imagewidth: int
) -> Tuple[int, int, int, int]:
    """Validate and clip a region to the image bounds.

    Returns ``(y0, x0, y1, x1)`` as absolute, in-bounds offsets.
    """

    try:
        y0, x0, y1, x1 = (int(v) for v in region)
    except (TypeError, ValueError):
        raise ValueError(
            "`region` must be a sequence of 4 ints (y0, x0, y1, x1), "
            f"got {region!r}."
        ) from None

    def _clip(value: int, dim: int) -> int:
        if value < 0:
            value += dim
        return min(max(value, 0), dim)

    y0, y1 = _clip(y0, imagelength), _clip(y1, imagelength)
    x0, x1 = _clip(x0, imagewidth), _clip(x1, imagewidth)

    if y0 >= y1 or x0 >= x1:
        raise ValueError(
            f"Region {region!r} is empty after clipping to the "
            f"image bounds (length={imagelength}, width={imagewidth})."
        )

    return y0, x0, y1, x1


def _slice_yx(
    ndimage: np.ndarray,
    axes: str,
    region: Tuple[int, int, int, int],
    imagelength: int,
    imagewidth: int,
) -> np.ndarray:
    """Slice the Y and X axes of ``ndimage`` to ``region``."""

    y0, x0, y1, x1 = _normalize_region(region, imagelength, imagewidth)
    index = [slice(None)] * ndimage.ndim
    index[axes.index("Y")] = slice(y0, y1)
    index[axes.index("X")] = slice(x0, x1)
    return ndimage[tuple(index)]


def _segment_positions(
    keyframe: "tifffile.TiffPage",
) -> Iterator[Tuple[int, int, int, int, int]]:
    """Yield ``(index, s, d, h, w)`` for every segment of a page.

    The position is given in samples of the normalized image shape
    ``(separate sample, depth, length, width, contig sample)`` and mirrors
    the index math of tifffile's own decoder closures.
    """

    samples, imdepth, imlength, imagewidth, _ = keyframe.shaped

    if keyframe.is_tiled:
        stdepth = keyframe.tiledepth
        stlength = keyframe.tilelength
        stwidth = keyframe.tilewidth
        width = math.ceil(imagewidth / stwidth)
        length = math.ceil(imlength / stlength)
        depth = math.ceil(imdepth / stdepth)
        area = width * length
        volume = area * depth
        for index in range(volume * samples):
            yield (
                index,
                index // volume,
                (index // area) % depth * stdepth,
                (index // width) % length * stlength,
                index % width * stwidth,
            )
    else:
        stdepth = 1
        stlength = keyframe.rowsperstrip
        length = math.ceil(imlength / stlength)
        for index in range(length * imdepth * samples):
            yield (
                index,
                index // (length * imdepth),
                (index // length) % imdepth * stdepth,
                index % length * stlength,
                0,
            )


def _tifffile_supports_partial_decode() -> bool:
    """Check if the installed tifffile exposes the segment-level decode API."""

    if not hasattr(tifffile.TiffPage, "decode"):
        return False
    if not hasattr(tifffile.FileHandle, "read_segments"):
        return False
    # read_segments must support the `flat` parameter (tifffile >= 2022.8)
    parameters = inspect.signature(tifffile.FileHandle.read_segments).parameters
    return "flat" in parameters


# evaluated once at import: tifffile's capabilities don't change at runtime
_TIFFFILE_SUPPORTS_PARTIAL_DECODE = _tifffile_supports_partial_decode()
# `_fullsize` was added to tifffile's decode closures together with
# TiffPage.segments; without it edge tiles decode padded, which the
# placement code in _read_page_region handles either way
_DECODE_HAS_FULLSIZE = (
    hasattr(tifffile.TiffPage, "segments")
    and "_fullsize" in inspect.signature(tifffile.TiffPage.segments).parameters
)


def _read_page_region(
    page: "tifffile.TiffPage", region: Tuple[int, int, int, int]
) -> np.ndarray:
    """Read a rectangular region from a single TIFF page.

    On files whose image data is split into multiple tiles or strips, only
    the segments intersecting the region are read from disk and decoded, so
    both I/O and memory scale with the region rather than the full page. On
    all other files (e.g. a single strip holding the whole image) this falls
    back to decoding the full page and slicing it.
    """

    keyframe = page.keyframe  # page may be a TiffFrame
    imagelength = keyframe.imagelength
    imagewidth = keyframe.imagewidth
    y0, x0, y1, x1 = _normalize_region(region, imagelength, imagewidth)

    n_segments = len(page.dataoffsets)
    can_decode_partially = (
        n_segments > 1
        and 0 not in keyframe.shaped
        and keyframe._dtype is not None
        and keyframe.jpegheader is None  # NDPI-style single JPEG stream
        and (keyframe.is_tiled or keyframe.rowsperstrip >= 1)
        and _TIFFFILE_SUPPORTS_PARTIAL_DECODE
    )

    if not can_decode_partially:
        return _slice_yx(
            page.asarray(), keyframe.axes, (y0, x0, y1, x1), imagelength, imagewidth
        )

    samples, imdepth, _, _, contig = keyframe.shaped
    if keyframe.is_tiled:
        stlength, stwidth = keyframe.tilelength, keyframe.tilewidth
    else:
        stlength, stwidth = keyframe.rowsperstrip, imagewidth

    # indices of segments intersecting the region
    selected = []
    for index, _, _, h, w in _segment_positions(keyframe):
        if h < y1 and h + stlength > y0 and w < x1 and w + stwidth > x0:
            selected.append(index)

    decodeargs: Dict[str, Any] = {}
    if _DECODE_HAS_FULLSIZE:
        decodeargs["_fullsize"] = False
    if keyframe.compression in {6, 7, 34892, 33007}:  # JPEG
        decodeargs["jpegtables"] = page.jpegtables
        decodeargs["jpegheader"] = keyframe.jpegheader
    decode = keyframe.decode

    fh = keyframe.parent.filehandle
    lock = fh.lock
    with lock:
        closed = fh.closed
        if closed:
            warnings.warn(
                f"{page!r} reading array from closed file",
                UserWarning,
                stacklevel=2,
            )
            fh.open()

    try:
        out = np.full(
            (samples, imdepth, y1 - y0, x1 - x0, contig),
            keyframe.nodata,
            dtype=keyframe._dtype,
        )

        offsets = [page.dataoffsets[i] for i in selected]
        bytecounts = [page.databytecounts[i] for i in selected]
        segments = fh.read_segments(
            offsets, bytecounts, length=len(selected), sort=True, flat=True, lock=lock
        )
        for data, position in segments:
            segment, (s, d, h, w, _), _ = decode(data, selected[position], **decodeargs)
            if segment is None:
                continue  # empty segment; out already holds nodata
            # edge tiles/strips may decode smaller than the nominal
            # segment shape; clip to the actual data and image bounds
            sd = min(segment.shape[0], imdepth - d)
            sh = min(segment.shape[1], imagelength - h)
            sw = min(segment.shape[2], imagewidth - w)
            oy0, oy1 = max(h, y0), min(h + sh, y1)
            ox0, ox1 = max(w, x0), min(w + sw, x1)
            if oy0 >= oy1 or ox0 >= ox1:
                continue
            out[s, d : d + sd, oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0] = segment[
                :sd, oy0 - h : oy1 - h, ox0 - w : ox1 - w
            ]
    finally:
        if closed:
            fh.close()

    # reshape the normalized (S, D, h, w, C) array to the page's shape,
    # with the Y and X axes cropped to the region
    axes = keyframe.axes
    shape = list(keyframe.shape)
    shape[axes.index("Y")] = y1 - y0
    shape[axes.index("X")] = x1 - x0
    return out.reshape(shape)


class TifffilePlugin(PluginV3):
    """Support for tifffile as backend.

    Parameters
    ----------
    request : iio.Request
        A request object that represents the users intent. It provides a
        standard interface for a plugin to access the various ImageResources.
        Check the docs for details.
    kwargs : Any
        Additional kwargs are forwarded to tifffile's constructor, i.e.
        to ``TiffFile`` for reading or ``TiffWriter`` for writing.

    """

    def __init__(self, request: Request, **kwargs) -> None:
        super().__init__(request)
        self._fh = None

        if request.mode.io_mode == "r":
            try:
                self._fh = tifffile.TiffFile(request.get_file(), **kwargs)
            except tifffile.tifffile.TiffFileError:
                raise InitializationError("Tifffile can not read this file.")
        else:
            self._fh = tifffile.TiffWriter(request.get_file(), **kwargs)

    # ---------------------
    # Standard V3 Interface
    # ---------------------

    def read(
        self,
        *,
        index: int = None,
        page: int = None,
        region: Tuple[int, int, int, int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Read a ndimage or page.

        The ndimage returned depends on the value of both ``index`` and
        ``page``. ``index`` selects the series to read and ``page`` allows
        selecting a single page from the selected series. If ``index=None``,
        ``page`` is understood as a flat index, i.e., the selection ignores
        individual series inside the file. If both ``index`` and ``page`` are
        ``None``, then all the series are read and returned as a batch.

        Parameters
        ----------
        index : int
            If ``int``, select the ndimage (series) located at that index inside
            the file and return ``page`` from it. If ``None`` and ``page`` is
            ``int`` read the page located at that (flat) index inside the file.
            If ``None`` and ``page=None``, read all ndimages from the file and
            return them as a batch.
        page : int
            If ``None`` return the full selected ndimage. If ``int``, read the
            page at the selected index and return it.
        region : tuple of int
            If given, read only the rectangular region ``(y0, x0, y1, x1)``
            of the image's Y and X axes and return an array of shape
            ``(..., y1 - y0, x1 - x0, ...)``. Coordinates follow numpy
            semantics: negative values count back from the corresponding
            image edge and out-of-bounds values are clipped to the image.
            A region that is empty after clipping raises a ``ValueError``.

            On files whose image data is split into multiple tiles or
            strips, only the segments intersecting the region are read from
            disk and decoded, so I/O and memory scale with the region size
            instead of the full page. On other files the full image is
            decoded and then sliced. The returned pixels are identical to
            reading the full image and slicing it. ``region`` cannot be
            combined with the tifffile-specific ``key`` or ``series``
            kwargs; other additional kwargs are ignored in region mode.
        kwargs : Any
            Additional kwargs are forwarded to TiffFile's ``as_array`` method.

        Returns
        -------
        ndarray : np.ndarray
            The decoded ndimage or page.
        """

        if region is not None:
            if "key" in kwargs or "series" in kwargs:
                raise ValueError("Can't use `region` together with `key` or `series`.")
            return self._read_region(region, index=index, page=page)

        if "key" not in kwargs:
            kwargs["key"] = page
        elif page is not None:
            raise ValueError("Can't use `page` and `key` at the same time.")

        # set plugin default for ``index``
        if index is not None and "series" in kwargs:
            raise ValueError("Can't use `series` and `index` at the same time.")
        elif "series" in kwargs:
            index = kwargs.pop("series")
        elif index is not None:
            pass
        else:
            index = 0

        if index is Ellipsis and page is None:
            # read all series in the file and return them as a batch
            ndimage = np.stack([x for x in self.iter(**kwargs)])
        else:
            index = None if index is Ellipsis else index
            ndimage = self._fh.asarray(series=index, **kwargs)

        return ndimage

    def iter(self, **kwargs) -> np.ndarray:
        """Yield ndimages from the TIFF.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs are forwarded to the TiffPageSeries' ``as_array``
            method.

        Yields
        ------
        ndimage : np.ndarray
            A decoded ndimage.
        """

        for sequence in self._fh.series:
            yield sequence.asarray(**kwargs)

    def _read_region(
        self,
        region: Tuple[int, int, int, int],
        *,
        index: int = None,
        page: int = None,
    ) -> np.ndarray:
        """Read a rectangular Y/X region of the selected ndimage or page.

        See :meth:`read` for the semantics of ``region``, ``index``, and
        ``page``.
        """

        if index is Ellipsis and page is None:
            # read the region from all series and return them as a batch
            return np.stack(
                [self._read_series_region(s, region) for s in self._fh.series]
            )

        if index is Ellipsis:
            # page is a flat index into the file's pages
            return _read_page_region(self._fh.pages[page], region)

        if index is None:
            index = 0
        series = self._fh.series[index]

        if page is not None:
            return _read_page_region(series.pages[page], region)

        return self._read_series_region(series, region)

    @staticmethod
    def _read_series_region(
        series: "tifffile.TiffPageSeries", region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Read a rectangular Y/X region from all pages of a series."""

        pages = series.pages
        if len(pages) == 1:
            return _read_page_region(pages[0], region)

        # Fast path: a homogeneous series is a plain stack of equally shaped
        # pages, so reading the region page-wise and stacking the results
        # yields the same array as slicing the stacked series.
        first = pages[0].keyframe
        is_plain_stack = tuple(series.shape) == (len(pages), *first.shape) and all(
            p.keyframe.shape == first.shape and p.keyframe.dtype == first.dtype
            for p in pages
        )
        if is_plain_stack:
            return np.stack([_read_page_region(p, region) for p in pages])

        # Fallback: decode the full series, then slice its Y and X axes.
        keyframe = series.pages[0].keyframe
        return _slice_yx(
            series.asarray(),
            series.axes,
            region,
            keyframe.imagelength,
            keyframe.imagewidth,
        )

    def write(
        self, ndimage: ArrayLike, *, is_batch: bool = False, **kwargs
    ) -> Optional[bytes]:
        """Save a ndimage as TIFF.

        Parameters
        ----------
        ndimage : ArrayLike
            The ndimage to encode and write to the ImageResource.
        is_batch : bool
            If True, the first dimension of the given ndimage is treated as a
            batch dimension and each element will create a new series.
        kwargs : Any
            Additional kwargs are forwarded to TiffWriter's ``write`` method.

        Returns
        -------
        encoded_image : bytes
            If the ImageResource is ``"<bytes>"``, return the encoded bytes.
            Otherwise write returns None.

        Notes
        -----
        Incremental writing is supported. Subsequent calls to ``write`` will
        create new series unless ``contiguous=True`` is used, in which case the
        call to write will append to the current series.

        When ``photometric`` / ``planarconfig`` are omitted, ImageIO sets
        defaults for common RGB layouts so tifffile does not emit deprecation
        warnings about ambiguous channel-axis interpretation:

        - channel-last arrays (shape ``(..., 3)`` or ``(..., 4)``):
          ``photometric="rgb"``
        - channel-first arrays (shape ``(3, ...)`` or ``(4, ...)`` on the
          third-to-last axis): ``photometric="rgb"`` and
          ``planarconfig="separate"``
        """

        if not is_batch:
            ndimage = np.asarray(ndimage)[None, :]

        for image in ndimage:
            write_kwargs = dict(kwargs)
            arr = np.asarray(image)
            if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
                write_kwargs.setdefault("photometric", "rgb")
            elif arr.ndim >= 3 and arr.shape[-3] in (3, 4):
                write_kwargs.setdefault("photometric", "rgb")
                write_kwargs.setdefault("planarconfig", "separate")
            self._fh.write(image, **write_kwargs)

        if self._request._uri_type == URI_BYTES:
            self._fh.close()
            file = cast(BytesIO, self._request.get_file())
            return file.getvalue()

    def metadata(
        self, *, index: int = Ellipsis, page: int = None, exclude_applied: bool = True
    ) -> Dict[str, Any]:
        """Format-Specific TIFF metadata.

        The metadata returned depends on the value of both ``index`` and
        ``page``. ``index`` selects a series and ``page`` allows selecting a
        single page from the selected series. If ``index=Ellipsis``, ``page`` is
        understood as a flat index, i.e., the selection ignores individual
        series inside the file. If ``index=Ellipsis`` and ``page=None`` then
        global (file-level) metadata is returned.

        Parameters
        ----------
        index : int
            Select the series of which to extract metadata from. If Ellipsis, treat
            page as a flat index into the file's pages.
        page : int
            If not None, select the page of which to extract metadata from. If
            None, read series-level metadata or, if ``index=...`` global,
            file-level metadata.
        exclude_applied : bool
            For API compatibility. Currently ignored.

        Returns
        -------
        metadata : dict
            A dictionary with information regarding the tiff flavor (file-level)
            or tiff tags (page-level).
        """

        if index is not Ellipsis and page is not None:
            target = self._fh.series[index].pages[page]
        elif index is not Ellipsis and page is None:
            # This is based on my understanding that series-level metadata is
            # stored in the first TIFF page.
            target = self._fh.series[index].pages[0]
        elif index is Ellipsis and page is not None:
            target = self._fh.pages[page]
        else:
            target = None

        metadata = {}
        if target is None:
            # return file-level metadata
            metadata["byteorder"] = self._fh.byteorder

            for flag in tifffile.TIFF.FILE_FLAGS:
                flag_value = getattr(self._fh, "is_" + flag)
                metadata["is_" + flag] = flag_value

                if flag_value and hasattr(self._fh, flag + "_metadata"):
                    flavor_metadata = getattr(self._fh, flag + "_metadata")
                    if isinstance(flavor_metadata, tuple):
                        metadata.update(flavor_metadata[0])
                    else:
                        metadata.update(flavor_metadata)
        else:
            # tifffile may return a TiffFrame instead of a page
            target = target.keyframe

            metadata.update({tag.name: tag.value for tag in target.tags})
            metadata.update(
                {
                    "planar_configuration": target.planarconfig,
                    "compression": target.compression,
                    "predictor": target.predictor,
                    "orientation": None,  # TODO
                    "description1": target.description1,
                    "description": target.description,
                    "software": target.software,
                    **_get_resolution(target),
                    "datetime": target.datetime,
                }
            )

        return metadata

    def properties(self, *, index: int = None, page: int = None) -> ImageProperties:
        """Standardized metadata.

        The properties returned depend on the value of both ``index`` and
        ``page``. ``index`` selects a series and ``page`` allows selecting a
        single page from the selected series. If ``index=Ellipsis``, ``page`` is
        understood as a flat index, i.e., the selection ignores individual
        series inside the file. If ``index=Ellipsis`` and ``page=None`` then
        global (file-level) properties are returned. If ``index=Ellipsis``
        and ``page=...``, file-level properties for the flattened index are
        returned.

        Parameters
        ----------
        index : int
            If ``int``, select the ndimage (series) located at that index inside
            the file. If ``Ellipsis`` and ``page`` is ``int`` extract the
            properties of the page located at that (flat) index inside the file.
            If ``Ellipsis`` and ``page=None``, return the properties for the
            batch of all ndimages in the file.
        page : int
            If ``None`` return the properties of the full ndimage. If ``...``
            return the properties of the flattened index. If ``int``,
            return the properties of the page at the selected index only.

        Returns
        -------
        image_properties : ImageProperties
            The standardized metadata (properties) of the selected ndimage or series.

        """
        index = index or 0
        page_idx = 0 if page in (None, Ellipsis) else page

        if index is Ellipsis:
            target_page = self._fh.pages[page_idx]
        else:
            target_page = self._fh.series[index].pages[page_idx]

        if index is Ellipsis and page is None:
            n_series = len(self._fh.series)
            props = ImageProperties(
                shape=(n_series, *target_page.shape),
                dtype=target_page.dtype,
                n_images=n_series,
                is_batch=True,
                spacing=_get_resolution(target_page).get("resolution"),
            )
        elif index is Ellipsis and page is Ellipsis:
            n_pages = len(self._fh.pages)
            props = ImageProperties(
                shape=(n_pages, *target_page.shape),
                dtype=target_page.dtype,
                n_images=n_pages,
                is_batch=True,
                spacing=_get_resolution(target_page).get("resolution"),
            )
        else:
            props = ImageProperties(
                shape=target_page.shape,
                dtype=target_page.dtype,
                is_batch=False,
                spacing=_get_resolution(target_page).get("resolution"),
            )

        return props

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()

        super().close()

    # ------------------------------
    # Add-on Interface inside imopen
    # ------------------------------

    def iter_pages(self, index=..., **kwargs):
        """Yield pages from a TIFF file.

        This generator walks over the flat index of the pages inside an
        ImageResource and yields them in order.

        Parameters
        ----------
        index : int
            The index of the series to yield pages from. If Ellipsis, walk over
            the file's flat index (and ignore individual series).
        kwargs : Any
            Additional kwargs are passed to TiffPage's ``as_array`` method.

        Yields
        ------
        page : np.ndarray
            A page stored inside the TIFF file.

        """

        if index is Ellipsis:
            pages = self._fh.pages
        else:
            pages = self._fh.series[index]

        for page in pages:
            yield page.asarray(**kwargs)
