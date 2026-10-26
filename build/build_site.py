from pathlib import Path
from bs4 import BeautifulSoup
import shutil

import markdown
from markdown.extensions.toc import slugify_unicode ##slugify_unicode bevarer æ, ø og å i overskrifternes id'er, 

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
SITE = ROOT / "site"

TEMPLATE = """<!doctype html>
<html lang="da">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>projektHireMe</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manufacturing+Consent&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="site.css">
  <script src="site.js" defer></script>
</head>
<body>
  <nav id="navbar" aria-label="Hovedmenu">
    <button type="button" class="nav-toggle" data-action="toggleTruncate" aria-pressed="true" title="Fold sammen / fold ud">
      <svg class="icon-on" viewBox="0 0 24 24" width="1.2em" height="1.2em" aria-hidden="true"><path d="M7 4l5 5 5-5M7 20l5-5 5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <svg class="icon-off" viewBox="0 0 24 24" width="1.2em" height="1.2em" aria-hidden="true"><path d="M7 9l5-5 5 5M7 15l5 5 5-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </button>
    <div class="nav-item">
      <button type="button" data-action="dropdown" aria-expanded="false">Placeholder</button>
      <div class="nav-dropdown">
        <p>Hold this place</p>
      </div>
    </div>
  </nav>
  <main>
    <article id="content">
{content}
    </article>
  </main>
</body>
</html>
"""


def main() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    html = markdown.markdown(
        text,
        extensions=["extra", "toc"],
        extension_configs={"toc": {"slugify": slugify_unicode}},
    )
    soup = BeautifulSoup(html, "html.parser")
    headings = ("h1", "h2", "h3", "h4", "h5")
    for heading in soup.find_all(headings):
        content = []
        for sibling in heading.find_next_siblings():
            if sibling.name == "hr":
                continue
            if sibling.name in headings:
                break
            content.append(sibling)
        if not content:
            continue
        wrap = soup.new_tag("div", attrs={"class": "truncate"})
        heading.insert_after(wrap)
        if sum(len(str(el)) for el in content) < 500:
            print(f"Skipping truncation for heading: {heading.text}")
            continue
        for el in content:
            wrap.append(el)  # append flytter elementet ind i div'en
    html = str(soup)
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    for f in SITE.iterdir():
        shutil.copy(f, DIST / f.name)
    (DIST / "index.html").write_text(TEMPLATE.format(content=html), encoding="utf-8")
    print(f"Skrev {DIST / 'index.html'}")


if __name__ == "__main__":
    main()