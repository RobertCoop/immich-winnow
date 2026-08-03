"""Thumbnail preparation and perceptual hashing.

Everything here is pure Pillow and fully deterministic: the same bytes in
always produce the same JPEG and the same hash, so cached thumbnails and
stored hashes stay stable across runs and across machines.
"""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageOps

__all__ = ["dhash", "hamming", "prepare_image", "to_b64"]


def prepare_image(raw: bytes, max_edge: int = 768, quality: int = 82) -> bytes:
    """Normalise arbitrary image bytes into a small, metadata-free JPEG.

    The image is EXIF-transposed (so the pixels match what a viewer shows),
    converted to RGB, and shrunk so that neither edge exceeds ``max_edge``.
    Images already at or below that size are never upscaled. Re-encoding
    drops every EXIF/XMP/ICC block, so nothing identifying is ever sent to
    the model.

    Args:
        raw: Encoded image bytes in any format Pillow can read.
        max_edge: Maximum length of the long edge, in pixels.
        quality: JPEG quality (1-100) for the re-encode.

    Returns:
        JPEG bytes.

    Raises:
        ValueError: If ``max_edge`` or ``quality`` is out of range.
        PIL.UnidentifiedImageError: If ``raw`` is not a readable image.
    """
    if max_edge < 1:
        raise ValueError(f"max_edge must be >= 1, got {max_edge}")
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be 1-100, got {quality}")

    buf = io.BytesIO()
    with Image.open(io.BytesIO(raw)) as img:
        oriented = ImageOps.exif_transpose(img) or img
        rgb = oriented.convert("RGB")
        # thumbnail() preserves aspect ratio and is a no-op when the image
        # already fits, which is exactly the "never upscale" rule.
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def to_b64(jpeg: bytes) -> str:
    """Base64-encode image bytes for an Anthropic image content block."""
    return base64.b64encode(jpeg).decode("ascii")


def dhash(raw: bytes, hash_size: int = 8) -> int:
    """Compute the classic difference hash of an image.

    The image is reduced to greyscale and resized to
    ``(hash_size + 1, hash_size)``; each bit records whether a pixel is
    brighter than its right-hand neighbour. Bits are packed row-major,
    most-significant bit first, giving a ``hash_size ** 2``-bit integer.

    Args:
        raw: Encoded image bytes in any format Pillow can read.
        hash_size: Width of the hash grid; 8 yields a 64-bit hash.

    Returns:
        The hash as a non-negative integer.

    Raises:
        ValueError: If ``hash_size`` is less than 1.
        PIL.UnidentifiedImageError: If ``raw`` is not a readable image.
    """
    if hash_size < 1:
        raise ValueError(f"hash_size must be >= 1, got {hash_size}")

    with Image.open(io.BytesIO(raw)) as img:
        small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    # "L" mode packs one byte per pixel, row-major, with no padding.
    pixels = small.tobytes()

    bits = 0
    stride = hash_size + 1
    for row in range(hash_size):
        base = row * stride
        for col in range(hash_size):
            bits = (bits << 1) | int(pixels[base + col] > pixels[base + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    """Return the number of differing bits between two hashes."""
    return (a ^ b).bit_count()
