"""
Export the HTML deck to a 16:9 PDF — bulletproof method.

Strategy:
  1. Launch headless Chromium at exactly 1600x900 (16:9)
  2. Use Reveal.js's JS API to navigate to each slide
  3. Screenshot each slide as a PNG at its native size
  4. Combine all PNGs into a single multi-page PDF with PIL

Each PDF page is exactly the same dimensions as the slide — no clipping,
no overflow, no print-CSS surprises.

Run:  python3 export_pdf.py
Output:  presentation.pdf  (and screenshots in /tmp/slide_*.png)
"""

import asyncio
import shutil
from pathlib import Path
from playwright.async_api import async_playwright
from PIL import Image

URL = "http://localhost:8000/presentation.html"
OUT_PDF = Path("presentation.pdf")
SLIDES_DIR = Path("/tmp/slides_export")
N_SLIDES = 10
WIDTH, HEIGHT = 1600, 900


async def main():
    if SLIDES_DIR.exists():
        shutil.rmtree(SLIDES_DIR)
    SLIDES_DIR.mkdir(parents=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=2,  # retina sharpness
        )
        page = await ctx.new_page()
        await page.goto(URL, wait_until="networkidle")
        # Let reveal.js + fonts settle
        await page.wait_for_timeout(2500)

        for i in range(N_SLIDES):
            # Jump to slide i (Reveal.js indices are 0-based)
            await page.evaluate(f"Reveal.slide({i}, 0)")
            await page.wait_for_timeout(800)  # transition + paint
            out_png = SLIDES_DIR / f"slide_{i:02d}.png"
            await page.screenshot(path=str(out_png), full_page=False,
                                  clip={"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT})
            print(f"captured slide {i+1}/{N_SLIDES} → {out_png}")

        await browser.close()

    # Combine all slide PNGs into a single PDF
    print("\nbuilding PDF ...")
    images = []
    for i in range(N_SLIDES):
        img = Image.open(SLIDES_DIR / f"slide_{i:02d}.png").convert("RGB")
        images.append(img)
    images[0].save(
        OUT_PDF, "PDF", resolution=150.0,
        save_all=True, append_images=images[1:],
    )
    print(f"saved → {OUT_PDF.resolve()}  ({OUT_PDF.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    asyncio.run(main())
