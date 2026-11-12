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
  <link rel="icon" type="image/svg+xml" href="favicon.svg">
    <meta name="description" content="Sådan brugte jeg RAG few-shot prompt-engineering og lokale sprog-modeller til at effektivisere mine jobansøgninger">
    <meta name="google-site-verification" content="QZ2IGpCfK4LfHnO70yYtOqGJFKeZ1ZYHyKpfhEXoPKA" />
    <link rel="canonical" href="https://nbkragh.github.io/projektHireMe/">
    <meta property="og:type" content="article">
    <meta property="og:title" content="projektHireMe – RAG til automatiserede jobansøgninger">
    <meta property="og:description" content="Et projekt med RAG, few-shot prompt-engineering og lokale sprog-modeller, Ollama">
    <meta property="og:url" content="https://nbkragh.github.io/projektHireMe/">
    <meta property="og:image" content="https://nbkragh.github.io/projektHireMe/img/scribes.jpg">
    <meta property="og:image:alt" content="Illustration og visualisering af autogenering af jobansøgninger">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Quintessential&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="site.css">
  <link rel="stylesheet" href="hal2026.css">
  <script src="site.js" defer></script>
</head>
<body>
  <nav id="navbar" aria-label="Hovedmenu">
    <button type="button" class="nav-toggle" data-action="toggleTruncate" aria-pressed="false" title="Fold sammen / fold ud">
      <svg class="icon-on" viewBox="0 0 24 24" width="1.2em" height="1.2em" aria-hidden="true"><path d="M7 4l5 5 5-5M7 20l5-5 5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <svg class="icon-off" viewBox="0 0 24 24" width="1.2em" height="1.2em" aria-hidden="true"><path d="M7 9l5-5 5 5M7 15l5 5 5-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </button>
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/nbkragh/projektHireMe" class="nav-item">github.com/nbkragh/projektHireMe</a>
    <div class="nav-item">
      <button type="button" data-action="dropdown" aria-expanded="false">Placeholder</button>
      <div class="nav-dropdown hal-dropdown">
        <div id="hal" class="circular">
          <div class="ring circular"></div>
          <div class="oculus circular">
            <div class="iris circular narrow"></div>
            <div class="iris circular wide"></div>
            <div class="highlight prime"></div>
            <div class="highlight secunda"></div>
          </div>
        </div>
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
        if len(content) == 0:
            continue
        wrap = soup.new_tag("div", attrs={"class": ["truncate"]})
        
        if sum(len(str(el)) for el in content) < 500:
            #print(f"Skipping truncation for heading: {heading.text}")
            continue
        for el in content:
            wrap.append(el)  # append flytter elementet ind i div'en
        heading.insert_after(wrap)
    html = str(soup)
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    for f in SITE.iterdir():
        if f.is_dir():
            shutil.copytree(f, DIST / f.name)
        else:
            shutil.copy(f, DIST / f.name)
    # README refererer til site/img/…, men i dist ligger billederne i img/
    html = html.replace("site/img/", "img/")
    (DIST / "index.html").write_text(TEMPLATE.format(content=html), encoding="utf-8")
    print(f"Skrev {DIST / 'index.html'}")


if __name__ == "__main__":
    main()