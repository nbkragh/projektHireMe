
// Egne handlers: navActions.mitNavn = (event, button) => {...}; knap i HTML: data-action="mitNavn"
const navActions = {
  toggleTruncate(e, btn) {
    const on = btn.getAttribute("aria-pressed") !== "true";
    btn.setAttribute("aria-pressed", on);
    document.querySelectorAll("#content .truncate").forEach((t) => t.classList.toggle("expanded", !on));
  },
  dropdown(e, btn) {
    const panel = btn.parentElement.querySelector(".nav-dropdown");
    if (!panel) return;
    const open = !panel.classList.contains("open");
    closeDropdowns();
    panel.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", open);
  },
};
window.navActions = navActions;

function closeDropdowns() {
  document.querySelectorAll(".nav-dropdown.open").forEach((p) => p.classList.remove("open"));
  document.querySelectorAll("#navbar [aria-expanded]").forEach((b) => b.setAttribute("aria-expanded", "false"));
}

document.addEventListener("DOMContentLoaded", () => {
  const nav = document.getElementById("navbar");
  nav.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const handler = navActions[btn.dataset.action];
      if (handler) handler(e, btn);
    });
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".nav-dropdown")) closeDropdowns(); });

  // opacity: 1 øverst og ved scroll op; falder gradvist ved scroll ned
  let lastY = window.scrollY, op = 1;
  const fade = 300; // px scroll for fuld fade
  window.addEventListener("scroll", () => {
    const y = window.scrollY, dy = y - lastY;
    lastY = y;
    if (y <= 0 || dy < 0) op = 1;
    else op = Math.max(0, op - dy / fade);
    nav.style.opacity = op;
    nav.classList.toggle("faded", op < 0.2);
  }, { passive: true });

  const root = document.getElementById("content");
  const truncs = root.querySelectorAll('[class="truncate"]');
  truncs.forEach((h) => h.addEventListener("click", () => h.classList.toggle("expanded")));


  // fold teksten ud, hvis man lander på et #anker på en h6
  const openHash = () => {
    const id = decodeURIComponent(location.hash.slice(1));
    const target = id && document.getElementById(id);
    if (target && target.tagName === "H5") target.classList.add("expanded");
  };
  window.addEventListener("hashchange", openHash);
  openHash();
});