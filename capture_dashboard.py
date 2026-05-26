"""
Capture screenshot(s) of the Streamlit dashboard for use in the presentation.

Outputs:
  images/dashboard_screenshot.png    — sentiment trajectory tab (for slide 9)
  images/dashboard_overview.png      — overview tab (alternative)
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

URL = "http://localhost:8501"
OUT_DIR = Path("images")
OUT_DIR.mkdir(exist_ok=True)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(
            viewport={"width": 1600, "height": 1000},
            device_scale_factor=2,  # retina-quality
        )
        page = await ctx.new_page()
        await page.goto(URL, wait_until="networkidle")

        # Give Streamlit a moment to finish rendering charts
        await page.wait_for_timeout(3000)

        # Overview tab is default — capture it first
        out1 = OUT_DIR / "dashboard_overview.png"
        await page.screenshot(path=str(out1), full_page=False)
        print(f"saved → {out1}")

        # Switch to Sentiment trajectory tab — most visually impressive
        # Streamlit tabs render as buttons; click by text
        try:
            await page.get_by_text("📈 Sentiment trajectory", exact=False).first.click()
            await page.wait_for_timeout(3500)  # allow plotly to render
            out2 = OUT_DIR / "dashboard_screenshot.png"
            await page.screenshot(path=str(out2), full_page=False)
            print(f"saved → {out2}")
        except Exception as e:
            print(f"could not switch tabs: {e}")
            # fallback: use the overview shot as the main one
            out2 = OUT_DIR / "dashboard_screenshot.png"
            await page.screenshot(path=str(out2), full_page=False)
            print(f"saved fallback → {out2}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
