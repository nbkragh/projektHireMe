from pathlib import Path
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
  <link rel="stylesheet" href="site.css">
  <script src="site.js" defer></script>
</head>
<body>
  <main>
    <div class="toolbar">
      <button type="button" id="expand-all">Fold alle ud</button>
      <button type="button" id="collapse-all">Fold alle sammen</button>
    </div>
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

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    for f in SITE.iterdir():
        shutil.copy(f, DIST / f.name)
    (DIST / "index.html").write_text(TEMPLATE.format(content=html), encoding="utf-8")
    print(f"Skrev {DIST / 'index.html'}")


if __name__ == "__main__":
    main()