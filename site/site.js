document.addEventListener("DOMContentLoaded", () => {
  const root = document.getElementById("content");
  const nodes = Array.from(root.children);
  root.innerHTML = "";

  // stack holder styr på, hvilken sektion nyt indhold hører under
  const stack = [{ level: 0, el: root }];

  for (const node of nodes) {
    const m = /^H([1-6])$/.exec(node.tagName);
    if (!m) {
      stack[stack.length - 1].el.appendChild(node);
      continue;
    }
    const level = Number(m[1]);
    while (stack[stack.length - 1].level >= level) stack.pop();

    const details = document.createElement("details");
    details.className = `section level-${level}`;
    details.open = level === 1;

    const summary = document.createElement("summary");
    summary.id = node.id; // så #ankre peger på summary
    node.removeAttribute("id");
    summary.appendChild(node);

    const body = document.createElement("div");
    body.className = "section-body";
    details.append(summary, body);

    stack[stack.length - 1].el.appendChild(details);
    stack.push({ level, el: body });
  }

  const setAll = (open) =>
    root.querySelectorAll("details").forEach((d) => (d.open = open));
  document.getElementById("expand-all").onclick = () => setAll(true);
  document.getElementById("collapse-all").onclick = () => setAll(false);

  // åbn forfædre, hvis man lander på/klikker et #anker i en lukket sektion
  const openHash = () => {
    const id = decodeURIComponent(location.hash.slice(1));
    const target = id && document.getElementById(id);
    if (!target) return;
    for (let d = target.closest("details"); d; d = d.parentElement.closest("details")) {
      d.open = true;
    }
    target.scrollIntoView();
  };
  window.addEventListener("hashchange", openHash);
  openHash();
});