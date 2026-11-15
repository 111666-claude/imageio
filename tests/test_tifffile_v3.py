"""Test tifffile plugin functionality."""

import datetime
import functools
import numpy as np
import pytest
import io
from copy import deepcopy
import struct

import imageio.v3 as iio
from imageio.config import known_plugins, known_extensions

tifffile = pytest.importorskip("tifffile", reason="TiffFile is not installed")


@pytest.fixture(scope="module", autouse=True)
def use_tifffile_v3():
    plugin_name = "tifffile"
    all_plugins = known_plugins.copy()
    all_extensions = known_extensions.copy()

    known_plugins.clear()
    known_extensions.clear()

    known_plugins[plugin_name] = all_plugins[plugin_name]

    for extension, configs in all_extensions.items():
        for config in configs:
            for plugin in config.priority:
                if plugin == plugin_name:
                    copied_config = deepcopy(config)
                    copied_config.priority = [plugin_name]
                    copied_config.default_priority = [plugin_name]
                    known_extensions[extension] = [copied_config]

    yield

    known_plugins.update(all_plugins)
    known_extensions.update(all_extensions)


def test_basic_roundtrip(tmp_path):
    filename = tmp_path / "test.tiff"
    expected = np.ones((10, 10, 3), np.uint8) * 2

    iio.imwrite(filename, expected)
    actual = iio.imread(filename)

    np.testing.assert_allclose(actual, expected)


def test_multiple_images_roundtrip(tmp_path):
    filename = tmp_path / "test.tiff"
    expected = np.ones((10, 10, 3), np.uint8) * 2

    iio.imwrite(filename, [expected] * 3, is_batch=True)

    # read all series as batch
    actual = iio.imread(filename, index=...)
    assert actual.shape == (3, 10, 10, 3)

    # read each series individually
    for idx in range(3):
        actual = iio.imread(filename, index=idx)
        np.testing.assert_allclose(actual, expected)


def test_volume_roundtrip(tmp_path):
    filename = tmp_path / "test.tiff"
    expected = np.ones((3, 10, 10, 3), np.uint8) * 2

    iio.imwrite(filename, expected)
    actual = iio.imread(filename)

    np.testing.assert_allclose(actual, expected)


def test_channel_first_volume(test_images, tmp_path):
    # remote channel-first volume rgb (2, 3, 10, 10)
    filename = test_images / "multipage_rgb.tif"
    img = iio.imread(filename)

    assert img.shape == (2, 3, 10, 10)


def test_planarconfig(tmp_path):
    filename = tmp_path / "test.tiff"
    expected = np.ones((3, 10, 10), np.uint8) * 2

    iio.imwrite(filename, expected, planarconfig="SEPARATE")

    # assert planar config
    with tifffile.TiffFile(filename) as tif:
        if tuple(int(x) for x in tifffile.__version__.split(".")) <= (2021, 11, 2):
            assert tif.series[0].pages[0].tags[284].value == 2
        else:
            # tifffile 2026.6.1+ (Python >= 3.12) uses axes codes; earlier
            # releases still report long dim names.
            assert tif.series[0].dims in (
                ("sample", "height", "width"),
                ("S", "Y", "X"),
            )


def test_metadata_reading(test_images):
    filename = test_images / "multipage_rgb.tif"

    file_metadata = iio.immeta(filename)
    assert file_metadata["is_imagej"] is False
    assert file_metadata["is_shaped"] is True

    page_metadata = iio.immeta(filename, index=0)
    assert page_metadata["description"] == "shape=(2,3,10,10)"
    assert page_metadata["description1"] == ""
    assert page_metadata["datetime"] == datetime.datetime(2015, 5, 9, 9, 8, 29)
    assert page_metadata["software"] == "tifffile.py"


def test_metadata_writing(tmp_path):
    filename = tmp_path / "test.tiff"
    dt = datetime.datetime(2018, 8, 6, 15, 35, 5)

    with iio.imopen(filename, "w") as file:
        file.write(
            np.zeros((10, 10)),
            software="testsoftware",
            description="test desc",
            datetime=dt,
        )
        file.write(np.zeros((10, 10)), description="another desc")

    page1_md = iio.immeta(filename, index=0)
    assert page1_md["datetime"] == dt
    assert page1_md["software"] == "testsoftware"
    assert page1_md["description"] == "test desc"

    page2_md = iio.immeta(filename, index=1)
    assert page2_md["description"] == "another desc"


def test_imagej_hyperstack(tmp_path):
    filename = tmp_path / "hyperstack.tiff"

    with iio.imopen(filename, "w", imagej=True) as file:
        file.write(np.zeros((15, 2, 180, 183), dtype=np.uint8))

    # test ImageIO plugin
    img = iio.imread(filename)
    assert img.shape == (15, 2, 180, 183)

    metadata = iio.immeta(filename)
    assert metadata["is_imagej"] is True


@pytest.mark.parametrize(
    "dpi,expected_resolution",
    [
        ((0, 1), (0, 1)),
        ((0, 12), (0, 12)),
        ((100, 200), (100, 200)),
        ((0.5, 0.5), (0.5, 0.5)),
        (((1, 3), (1, 3)), (1 / 3, 1 / 3)),
    ],
)
def test_resolution_metadata(tmp_path, dpi, expected_resolution):
    filename = tmp_path / "test.tif"
    data = np.zeros((200, 100), dtype=np.uint8)

    iio.imwrite(filename, data, resolution=dpi)
    page_metadata = iio.immeta(filename, index=0)

    assert page_metadata["resolution"] == expected_resolution
    assert page_metadata["resolution_unit"] == 2


@pytest.mark.parametrize("resolution", [(1, 0), (0, 0)])
def test_invalid_resolution_metadata(tmp_path, resolution):
    tif_path = tmp_path / "test.tif"
    data = np.zeros((200, 100), dtype=np.uint8)
    iio.imwrite(tif_path, data)

    with tifffile.TiffFile(tif_path, mode="r+b") as tif:
        tags = tif.pages[0].tags
        tags["XResolution"].overwrite(resolution)
        tags["YResolution"].overwrite(resolution)

    with pytest.warns(RuntimeWarning):
        read_image = iio.immeta(tmp_path / "test.tif", index=0)
    assert read_image["XResolution"] == resolution
    assert read_image["YResolution"] == resolution
    assert "resolution" not in read_image


def test_missing_resolution_metadata(tmp_path):
    tif_path = tmp_path / "test.tif"
    data = np.zeros((200, 100), dtype=np.uint8)
    iio.imwrite(tif_path, data)

    with tifffile.TiffFile(tif_path, mode="r+b") as tif:
        tags = tif.pages[0].tags
        xres = tags["XResolution"]
        tags["YResolution"].overwrite((10, 11))

    with tif_path.open("r+b") as f:
        # give XResolution tag another ID so it won't be recognized
        f.seek(xres.offset)
        f.write(struct.pack("<H", 65000))

    read_image = iio.immeta(tif_path, index=0)
    assert "XResolution" not in read_image
    assert read_image["YResolution"] == (10, 11)
    assert "resolution" not in read_image
    assert read_image["resolution_unit"] == 1

    tif_path2 = tmp_path / "test2.tif"
    data2 = np.zeros((200, 100), dtype=np.uint8)
    iio.imwrite(tif_path2, data2)

    with tifffile.TiffFile(tif_path2, mode="r+b") as tif:
        tags = tif.pages[0].tags
        res_u = tags["ResolutionUnit"]

    with tif_path2.open("r+b") as f:
        # give ResolutionUnit tag another ID so it won't be recognized
        f.seek(res_u.offset)
        f.write(struct.pack("<H", 65000))

    read_image2 = iio.immeta(tif_path2, index=0)
    assert "ResolutionUnit" not in read_image2
    assert "resolution" not in read_image2
    assert "resolution_unit" not in read_image2


def test_read_bytes():
    # regression test for: https://github.com/imageio/imageio/issues/703

    some_bytes = iio.imwrite("<bytes>", [[0]])
    assert some_bytes is not None


def test_tiff_page_writing():
    # regression test for
    # https://github.com/imageio/imageio/issues/849
    base_image = np.full((256, 256, 3), 42, dtype=np.uint8)

    buffer = io.BytesIO()
    iio.imwrite(buffer, base_image, extension=".tiff")
    buffer.seek(0)

    with tifffile.TiffFile(buffer) as file:
        assert len(file.pages) == 1


def test_bool_writing():
    # regression test for
    # https://github.com/imageio/imageio/issues/852

    expected = (np.arange(255 * 123) % 2 == 0).reshape((255, 123))

    img_bytes = iio.imwrite("<bytes>", expected, extension=".tiff")
    actual = iio.imread(img_bytes)

    assert np.allclose(actual, expected)


def test_roundtrip(tmp_path):
    # regression test for
    # https://github.com/imageio/imageio/issues/854

    iio.imwrite(tmp_path / "test.tiff", np.ones((10, 64, 64), "u4"))
    actual = iio.imread(tmp_path / "test.tiff")

    assert actual.shape == (10, 64, 64)


def test_volume_roudtrip(tmp_path):
    # regression test for
    # https://github.com/imageio/imageio/issues/818

    expected_volume = np.full((23, 123, 456, 3), 42, dtype=np.uint8)
    iio.imwrite(tmp_path / "volume.tiff", expected_volume)

    # assert that the file indeed contains a volume
    with tifffile.TiffFile(tmp_path / "volume.tiff") as file:
        assert file.series[0].shape == (23, 123, 456, 3)
        assert len(file.series) == 1

    actual_volume = iio.imread(tmp_path / "volume.tiff")
    assert np.allclose(actual_volume, expected_volume)


def test_multipage_read(tmp_path):
    # regression test for
    # https://github.com/imageio/imageio/issues/818

    filename = tmp_path / "flat.tiff"

    # this creates a TIFF with two flat images (non-volumetric)
    expected_flat = np.full((35, 73, 3), 114, dtype=np.uint8)
    iio.imwrite(filename, [expected_flat, expected_flat], is_batch=True)

    actual_flat = iio.imread(filename)
    assert np.allclose(actual_flat, expected_flat)

    for idx, series in enumerate(iio.imiter(filename)):
        assert np.allclose(series, expected_flat)
    assert idx == 1


def test_multiple_ndimages(tmp_path):
    # regression test for
    # https://github.com/imageio/imageio/issues/81

    filename = tmp_path / "nightmare.tiff"
    volumetric = np.full((4, 255, 255, 3), 114, dtype=np.uint8)
    flat = np.full((255, 255, 3), 114, dtype=np.uint8)
    different_shape = np.full((120, 73, 3), 114, dtype=np.uint8)
    with iio.imopen(filename, "w") as file:
        file.write(volumetric)
        file.write(flat)
        file.write(different_shape)

    # imread will read the respective series when using series kwarg
    assert iio.imread(filename, index=0).shape == (4, 255, 255, 3)
    assert iio.imread(filename, index=1).shape == (255, 255, 3)
    assert iio.imread(filename, index=2).shape == (120, 73, 3)

    # imread will read the respective plane when using index kwarg
    assert iio.imread(filename, index=..., page=0).shape == (255, 255, 3)
    assert iio.imread(filename, index=..., page=1).shape == (255, 255, 3)
    assert iio.imread(filename, index=..., page=2).shape == (255, 255, 3)
    assert iio.imread(filename, index=..., page=3).shape == (255, 255, 3)
    assert iio.imread(filename, index=..., page=4).shape == (255, 255, 3)
    assert iio.imread(filename, index=..., page=5).shape == (120, 73, 3)

    # imiter will yield the three images in order
    shapes = [(4, 255, 255, 3), (255, 255, 3), (120, 73, 3)]
    for image, shape in zip(iio.imiter(filename), shapes):
        assert image.shape == shape

    # iter pages will yield pages in order
    shapes = [
        (255, 255, 3),
        (255, 255, 3),
        (255, 255, 3),
        (255, 255, 3),
        (255, 255, 3),
        (120, 73, 3),
    ]
    with iio.imopen(filename, "r") as file:
        for image, shape in zip(file.iter_pages(), shapes):
            assert image.shape == shape


def test_compression(tmp_path):
    img = np.ones((128, 128))

    iio.imwrite(tmp_path / "test.tiff", img, compression="zlib")
    page_meta = iio.immeta(tmp_path / "test.tiff", index=0)
    assert page_meta["compression"] == 8

    iio.imwrite(tmp_path / "test2.tiff", img)
    page_meta = iio.immeta(tmp_path / "test2.tiff", index=0)
    assert page_meta["compression"] == 1


def test_properties(tmp_path, test_images):
    filename = tmp_path / "test.tiff"
    data = np.full((255, 255, 3), 42, dtype=np.uint8)
    iio.imwrite(filename, data)

    props = iio.improps(filename)
    assert props.shape == (255, 255, 3)
    assert props.n_images is None

    props = iio.improps(filename, index=...)
    assert props.shape == (1, 255, 255, 3)
    assert props.n_images == 1

    data = np.full((6, 255, 255, 3), 42, dtype=np.uint8)
    iio.imwrite(filename, data)

    props = iio.improps(filename, page=3)
    assert props.shape == (255, 255, 3)
    assert props.n_images is None

    props = iio.improps(filename, index=...)
    assert props.shape == (1, 255, 255, 3)
    assert props.n_images == 1

    props = iio.improps(filename, index=..., page=...)
    assert props.shape == (6, 255, 255, 3)
    assert props.n_images == 6

    # read file without resolution tags
    props = iio.improps(test_images / "multipage_rgb.tif")
    assert props.spacing is None


def test_contigous_writing(tmp_path):
    filename = tmp_path / "test.tiff"
    data = np.full((128, 128, 3), 42, np.uint8)

    with tifffile.TiffWriter(filename) as tw:
        tw.write(data, contiguous=True)
        tw.write(data, contiguous=True)
        tw.write(data, contiguous=True)
        tw.write(data, contiguous=True)

    metadata = iio.immeta(filename)
    assert metadata["is_shaped"] is True
    assert metadata["shape"] == [4, 128, 128, 3]


def test_unreadable_file():
    garbage = b"awdadbj30993urfj"

    with pytest.raises(OSError):
        iio.imread(garbage)


def test_touch_invalid_kwargs(tmp_path):
    filename = tmp_path / "test.tiff"
    data = np.full((5, 128, 128, 3), 42, np.uint8)
    iio.imwrite(filename, data)

    with pytest.raises(ValueError):
        iio.imread(filename, key=0, page=0)

    with pytest.raises(ValueError):
        iio.imread(filename, series=0, index=0)


def test_tiffile_native_arguments(tmp_path):
    filename = tmp_path / "test.tiff"
    data = np.full((5, 128, 128, 3), 42, np.uint8)
    data[0, 2, 3, :] = 7
    iio.imwrite(filename, data)

    page = iio.imread(filename, key=0)
    np.testing.assert_allclose(page, data[0])

    img = iio.imread(filename, series=0)
    np.testing.assert_allclose(img, data)


def test_various_metadata_reading(tmp_path):
    filename = tmp_path / "nightmare.tiff"
    flat = np.full((255, 255, 3), 114, dtype=np.uint8)
    volumetric = np.full((4, 255, 255, 3), 114, dtype=np.uint8)
    different_shape = np.full((120, 73, 3), 114, dtype=np.uint8)
    with iio.imopen(filename, "w") as file:
        file.write(flat)
        file.write(volumetric, compression="zlib")
        file.write(different_shape)

    metadata = iio.immeta(filename, index=1, page=2)
    assert metadata["compression"] == 8  # (ADOBE_DEFLATE)

    metadata = iio.immeta(filename, index=..., page=2)
    assert metadata["compression"] == 8  # (ADOBE_DEFLATE)


def test_iter_pages(tmp_path):
    filename = tmp_path / "nightmare.tiff"
    flat = np.full((255, 255, 3), 114, dtype=np.uint8)
    volumetric = np.full((4, 255, 255, 3), 114, dtype=np.uint8)
    different_shape = np.full((120, 73, 3), 114, dtype=np.uint8)
    with iio.imopen(filename, "w") as file:
        file.write(flat)
        file.write(volumetric, compression="zlib")
        file.write(different_shape)

    with iio.imopen(filename, "r") as file:
        for idx, page in enumerate(file.iter_pages(index=1)):
            np.testing.assert_allclose(page, volumetric[idx])


@pytest.fixture()
def region_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (517, 803, 3), np.uint8)


def test_region_read_tiled(tmp_path, region_image):
    filename = tmp_path / "tiled.tiff"
    iio.imwrite(filename, region_image, tile=(128, 128))

    full = iio.imread(filename)
    for region in [
        (0, 0, 517, 803),  # full image
        (300, 500, 517, 803),  # bottom-right corner (partial edge tiles)
        (0, 0, 1, 1),  # single pixel
        (128, 128, 256, 256),  # exactly one tile
        (100, 100, 101, 101),  # 1x1 inside a tile
        (64, 64, 700, 900),  # clipped to bounds
        (-100, -50, 10**6, 10**6),  # negative + out-of-bounds
    ]:
        y0, x0, y1, x1 = region
        y0 = y0 + 517 if y0 < 0 else y0
        x0 = x0 + 803 if x0 < 0 else x0
        expected = full[max(y0, 0) : min(y1, 517), max(x0, 0) : min(x1, 803)]
        actual = iio.imread(filename, region=region)
        np.testing.assert_array_equal(actual, expected)


def test_region_read_strips(tmp_path, region_image):
    filename = tmp_path / "strips.tiff"
    iio.imwrite(filename, region_image, rowsperstrip=37)

    full = iio.imread(filename)
    actual = iio.imread(filename, region=(50, 60, 510, 800))
    np.testing.assert_array_equal(actual, full[50:510, 60:800])


def test_region_read_compressed(tmp_path, region_image):
    filename = tmp_path / "compressed.tiff"
    iio.imwrite(
        filename, region_image, tile=(128, 128), compression="deflate", predictor=2
    )

    full = iio.imread(filename)
    actual = iio.imread(filename, region=(33, 65, 500, 800))
    np.testing.assert_array_equal(actual, full[33:500, 65:800])


def test_region_read_jpeg(tmp_path, region_image):
    pytest.importorskip("imagecodecs")
    filename = tmp_path / "jpeg.tiff"
    iio.imwrite(filename, region_image, tile=(256, 256), compression="jpeg")

    full = iio.imread(filename)
    actual = iio.imread(filename, region=(100, 200, 400, 700))
    np.testing.assert_array_equal(actual, full[100:400, 200:700])


def test_region_read_single_strip_fallback(tmp_path, region_image):
    # a single strip spans the whole image and cannot be decoded partially;
    # the region read must still be pixel-identical to slicing a full read
    filename = tmp_path / "single_strip.tiff"
    iio.imwrite(filename, region_image)

    full = iio.imread(filename)
    actual = iio.imread(filename, region=(10, 20, 500, 800))
    np.testing.assert_array_equal(actual, full[10:500, 20:800])


def test_region_read_multipage(tmp_path):
    rng = np.random.default_rng(13)
    pages = [rng.integers(0, 255, (200, 300), np.uint8) for _ in range(3)]

    filename = tmp_path / "multipage.tiff"
    iio.imwrite(filename, pages, is_batch=True, contiguous=True, rowsperstrip=48)

    region = (10, 20, 100, 200)
    y0, x0, y1, x1 = region

    # individual pages of the series
    for i, page in enumerate(pages):
        actual = iio.imread(filename, page=i, region=region)
        np.testing.assert_array_equal(actual, page[y0:y1, x0:x1])

    # the whole series
    actual = iio.imread(filename, index=0, region=region)
    np.testing.assert_array_equal(actual, np.stack(pages)[:, y0:y1, x0:x1])

    # all series as a batch
    actual = iio.imread(filename, index=..., region=region)
    full_batch = iio.imread(filename, index=...)
    np.testing.assert_array_equal(actual, full_batch[:, :, y0:y1, x0:x1])

    # flat page index
    actual = iio.imread(filename, index=..., page=2, region=region)
    np.testing.assert_array_equal(actual, pages[2][y0:y1, x0:x1])


def test_region_read_planar(tmp_path, region_image):
    filename = tmp_path / "planar.tiff"
    tifffile.imwrite(
        filename,
        np.moveaxis(region_image, -1, 0),
        planarconfig="separate",
        photometric="rgb",
        rowsperstrip=100,
    )

    full = iio.imread(filename)  # shape (samples, Y, X)
    actual = iio.imread(filename, region=(30, 40, 500, 700))
    np.testing.assert_array_equal(actual, full[:, 30:500, 40:700])


def test_region_read_uint16(tmp_path):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 65535, (300, 300), np.uint16)

    filename = tmp_path / "uint16.tiff"
    iio.imwrite(filename, image, tile=(64, 64))

    actual = iio.imread(filename, region=(33, 65, 299, 299))
    np.testing.assert_array_equal(actual, image[33:299, 65:299])


def test_region_read_empty_raises(tmp_path, region_image):
    filename = tmp_path / "tiled.tiff"
    iio.imwrite(filename, region_image, tile=(128, 128))

    with pytest.raises(ValueError, match="empty"):
        iio.imread(filename, region=(10, 10, 5, 20))

    with pytest.raises(ValueError, match="empty"):
        # fully outside the image
        iio.imread(filename, region=(10**6, 0, 10**7, 10))

    with pytest.raises(ValueError, match="4 ints"):
        iio.imread(filename, region=(0, 0, 10))


def test_region_read_conflicting_kwargs(tmp_path, region_image):
    filename = tmp_path / "tiled.tiff"
    iio.imwrite(filename, region_image, tile=(128, 128))

    with pytest.raises(ValueError, match="region"):
        iio.imread(filename, region=(0, 0, 10, 10), key=0)

    with pytest.raises(ValueError, match="region"):
        iio.imread(filename, region=(0, 0, 10, 10), series=0)


def test_region_read_io_scales_with_region(tmp_path, monkeypatch):
    # On chunked files, a small region read must only read the intersecting
    # tiles from disk, not the whole page.
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, (2048, 2048), np.uint8)

    filename = tmp_path / "big_tiled.tiff"
    iio.imwrite(filename, image, tile=(256, 256), compression="deflate")

    bytes_read = 0
    original_read_segments = tifffile.FileHandle.read_segments

    @functools.wraps(original_read_segments)
    def counting_read_segments(self, offsets, bytecounts, /, *args, **kwargs):
        nonlocal bytes_read
        bytes_read += sum(bytecounts)
        yield from original_read_segments(self, offsets, bytecounts, *args, **kwargs)

    monkeypatch.setattr(tifffile.FileHandle, "read_segments", counting_read_segments)

    iio.imread(filename)
    full_bytes = bytes_read

    bytes_read = 0
    region = iio.imread(filename, region=(1024, 1024, 1280, 1280))
    region_bytes = bytes_read

    np.testing.assert_array_equal(region, image[1024:1280, 1024:1280])
    # the 256x256 region covers 1/64 of the image and intersects exactly
    # one 256x256 tile; allow generous slack for tag/IFD re-reads
    assert region_bytes < full_bytes // 8
