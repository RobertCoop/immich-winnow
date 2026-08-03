"""Tests for winnow.images — thumbnail preparation and perceptual hashing."""

from __future__ import annotations

import base64
import io
import random

import pytest
from PIL import Image, UnidentifiedImageError

from winnow.images import dhash, hamming, prepare_image, to_b64

ORIENTATION_TAG = 274


def encode(img: Image.Image, fmt: str = "JPEG", **kwargs: object) -> bytes:
    """Encode a PIL image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def solid(width: int, height: int, color: tuple[int, int, int] = (200, 30, 90)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def ramp(width: int = 256, height: int = 64, reverse: bool = False) -> Image.Image:
    """Horizontal linear greyscale ramp (dark -> light, or reversed)."""
    img = Image.new("L", (width, height))
    row = [int(255 * x / (width - 1)) for x in range(width)]
    if reverse:
        row = [255 - v for v in row]
    img.putdata(row * height)
    return img


def blocks(seed: int, size: int = 8, scale: int = 32) -> Image.Image:
    """A seeded blocky greyscale pattern, scaled up with nearest-neighbour."""
    rng = random.Random(seed)
    small = Image.new("L", (size, size))
    small.putdata([rng.randrange(256) for _ in range(size * size)])
    return small.resize((size * scale, size * scale), Image.Resampling.NEAREST)


# --------------------------------------------------------------------------
# prepare_image
# --------------------------------------------------------------------------


def test_prepare_image_returns_jpeg(jpeg_bytes: bytes) -> None:
    out = prepare_image(jpeg_bytes)
    assert out[:2] == b"\xff\xd8"
    with Image.open(io.BytesIO(out)) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"


def test_prepare_image_downscales_long_edge_preserving_aspect() -> None:
    raw = encode(solid(2000, 1000))
    with Image.open(io.BytesIO(prepare_image(raw))) as img:
        assert img.size == (768, 384)


def test_prepare_image_respects_max_edge_argument() -> None:
    raw = encode(solid(1000, 2000))
    with Image.open(io.BytesIO(prepare_image(raw, max_edge=200))) as img:
        assert img.size == (100, 200)


def test_prepare_image_never_upscales() -> None:
    raw = encode(solid(120, 90))
    with Image.open(io.BytesIO(prepare_image(raw, max_edge=768))) as img:
        assert img.size == (120, 90)


def test_prepare_image_exact_fit_is_unchanged_in_size() -> None:
    raw = encode(solid(768, 768))
    with Image.open(io.BytesIO(prepare_image(raw))) as img:
        assert img.size == (768, 768)


def test_prepare_image_converts_rgba_to_rgb() -> None:
    raw = encode(Image.new("RGBA", (64, 64), (10, 20, 30, 128)), fmt="PNG")
    with Image.open(io.BytesIO(prepare_image(raw))) as img:
        assert img.mode == "RGB"
        assert img.format == "JPEG"


def test_prepare_image_converts_palette_to_rgb() -> None:
    raw = encode(solid(64, 64).convert("P"), fmt="PNG")
    with Image.open(io.BytesIO(prepare_image(raw))) as img:
        assert img.mode == "RGB"


def test_prepare_image_converts_greyscale_to_rgb() -> None:
    raw = encode(ramp(120, 80))
    with Image.open(io.BytesIO(prepare_image(raw))) as img:
        assert img.mode == "RGB"


def test_prepare_image_applies_exif_transpose() -> None:
    # Left half red, right half blue; orientation 6 => rotate 90 clockwise,
    # so the left column must end up as the top row.
    img = solid(200, 100, (255, 0, 0))
    img.paste(Image.new("RGB", (100, 100), (0, 0, 255)), (100, 0))
    exif = img.getexif()
    exif[ORIENTATION_TAG] = 6
    raw = encode(img, exif=exif)

    with Image.open(io.BytesIO(prepare_image(raw))) as out:
        assert out.size == (100, 200)  # dimensions swapped
        top = out.getpixel((50, 25))
        bottom = out.getpixel((50, 175))
    assert top[0] > top[2]  # top is red
    assert bottom[2] > bottom[0]  # bottom is blue


def test_prepare_image_strips_metadata() -> None:
    img = solid(300, 200)
    exif = img.getexif()
    exif[ORIENTATION_TAG] = 1
    exif[271] = "TestCam"  # Make
    exif[272] = "SecretModel"  # Model
    raw = encode(img, exif=exif)
    assert b"SecretModel" in raw

    out = prepare_image(raw)
    assert b"SecretModel" not in out
    with Image.open(io.BytesIO(out)) as prepared:
        assert len(prepared.getexif()) == 0
        assert "exif" not in prepared.info


def test_prepare_image_quality_affects_size() -> None:
    raw = encode(blocks(7, size=32, scale=16), quality=95)
    assert len(prepare_image(raw, quality=20)) < len(prepare_image(raw, quality=95))


def test_prepare_image_is_deterministic(jpeg_bytes: bytes) -> None:
    assert prepare_image(jpeg_bytes) == prepare_image(jpeg_bytes)


@pytest.mark.parametrize("max_edge", [0, -5])
def test_prepare_image_rejects_bad_max_edge(jpeg_bytes: bytes, max_edge: int) -> None:
    with pytest.raises(ValueError, match="max_edge"):
        prepare_image(jpeg_bytes, max_edge=max_edge)


@pytest.mark.parametrize("quality", [0, 101])
def test_prepare_image_rejects_bad_quality(jpeg_bytes: bytes, quality: int) -> None:
    with pytest.raises(ValueError, match="quality"):
        prepare_image(jpeg_bytes, quality=quality)


def test_prepare_image_rejects_non_image_bytes() -> None:
    with pytest.raises(UnidentifiedImageError):
        prepare_image(b"definitely not an image")


# --------------------------------------------------------------------------
# to_b64
# --------------------------------------------------------------------------


def test_to_b64_round_trips(jpeg_bytes: bytes) -> None:
    encoded = to_b64(jpeg_bytes)
    assert isinstance(encoded, str)
    assert encoded.isascii()
    assert base64.b64decode(encoded) == jpeg_bytes


def test_to_b64_empty() -> None:
    assert to_b64(b"") == ""


# --------------------------------------------------------------------------
# dhash
# --------------------------------------------------------------------------


def test_dhash_is_deterministic(jpeg_bytes: bytes) -> None:
    assert dhash(jpeg_bytes) == dhash(jpeg_bytes)


def test_dhash_solid_colour_is_zero() -> None:
    # Every pixel equals its neighbour, so no bit is ever set.
    assert dhash(encode(solid(320, 240))) == 0


def test_dhash_increasing_ramp_is_all_zeros() -> None:
    assert dhash(encode(ramp())) == 0


def test_dhash_decreasing_ramp_is_all_ones() -> None:
    assert dhash(encode(ramp(reverse=True))) == (1 << 64) - 1


def test_dhash_fits_in_hash_size_squared_bits(jpeg_bytes: bytes) -> None:
    raw = encode(blocks(3))
    assert dhash(raw).bit_length() <= 64
    assert dhash(raw, hash_size=4).bit_length() <= 16
    assert dhash(raw, hash_size=16).bit_length() <= 256
    assert dhash(jpeg_bytes) >= 0


def test_dhash_hash_size_one() -> None:
    value = dhash(encode(ramp()), hash_size=1)
    assert value in (0, 1)


def test_dhash_distinguishes_different_images() -> None:
    a = dhash(encode(blocks(1)))
    b = dhash(encode(blocks(2)))
    assert hamming(a, b) >= 10


def test_dhash_is_roughly_scale_invariant() -> None:
    small = dhash(encode(blocks(11, scale=16)))
    large = dhash(encode(blocks(11, scale=64)))
    assert hamming(small, large) <= 6


def test_dhash_survives_recompression() -> None:
    original = encode(blocks(5), quality=95)
    assert hamming(dhash(original), dhash(prepare_image(original, max_edge=256))) <= 6


def test_dhash_accepts_png() -> None:
    png = encode(blocks(9), fmt="PNG")
    jpeg = encode(blocks(9), quality=95)
    assert hamming(dhash(png), dhash(jpeg)) <= 4


@pytest.mark.parametrize("hash_size", [0, -1])
def test_dhash_rejects_bad_hash_size(jpeg_bytes: bytes, hash_size: int) -> None:
    with pytest.raises(ValueError, match="hash_size"):
        dhash(jpeg_bytes, hash_size=hash_size)


def test_dhash_rejects_non_image_bytes() -> None:
    with pytest.raises(UnidentifiedImageError):
        dhash(b"nope")


# --------------------------------------------------------------------------
# hamming
# --------------------------------------------------------------------------


def test_hamming_identical_is_zero() -> None:
    assert hamming(0, 0) == 0
    assert hamming(0xDEADBEEF, 0xDEADBEEF) == 0


def test_hamming_counts_differing_bits() -> None:
    assert hamming(0b0000, 0b1011) == 3
    assert hamming(0b1111, 0b0000) == 4
    assert hamming(1, 2) == 2


def test_hamming_is_symmetric() -> None:
    a, b = 0x0F0F0F0F, 0x00FF00FF
    assert hamming(a, b) == hamming(b, a)


def test_hamming_full_width() -> None:
    assert hamming(0, (1 << 64) - 1) == 64
