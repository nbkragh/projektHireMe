
// Egne handlers: navActions.mitNavn = (event, button) => {...}; knap i HTML: data-action="mitNavn"
const navActions = {
  toggleTruncate(e, btn) {
    const on = btn.getAttribute("aria-pressed") !== "true";
    btn.setAttribute("aria-pressed", on);
    document.querySelectorAll("#content .truncate").forEach((t) => t.classList.toggle("expanded", on));
  },
  dropdown(e, btn) {
    const panel = btn.parentElement.querySelector(".nav-dropdown");
    if (!panel) return;
    const open = !panel.classList.contains("open");
    closeDropdowns();
    panel.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", open);
    const eye = panel.querySelector("#hal");
    if (open && eye && lastEyePoint) lookAt(eye, lastEyePoint.x, lastEyePoint.y);
  },
};
window.navActions = navActions;

function closeDropdowns() {
  document.querySelectorAll(".nav-dropdown.open").forEach((p) => p.classList.remove("open"));
  document.querySelectorAll("#navbar [aria-expanded]").forEach((b) => b.setAttribute("aria-expanded", "false"));
}

let lastEyePoint = null;

function lookAt(eye, clientX, clientY) {
  lastEyePoint = { x: clientX, y: clientY };
  const rect = eye.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const normalize = (point, center, size) => {
    center = Math.max(0, Math.min(size, center));
    point = Math.max(0, Math.min(size, point));
    const distance = point - center;
    const extent = distance < 0 ? center : size - center;
    return extent ? distance / extent : 0;
  };
  const x = normalize(clientX, rect.left + rect.width / 2, window.innerWidth);
  const y = normalize(clientY, rect.top + rect.height / 2, window.innerHeight);
  // Map the viewport rectangle to a circle without flattening its corners.
  const circleX = x * Math.sqrt(1 - y * y / 2);
  const circleY = y * Math.sqrt(1 - x * x / 2);
  eye.style.setProperty("--iris-x", `${50 + 50 * circleX}%`);
  eye.style.setProperty("--iris-y", `${50 + 50 * circleY}%`);
}

document.addEventListener("DOMContentLoaded", () => {
  const nav = document.getElementById("navbar");
  let touchDrag = null;
  let completedDrag = null;
  document.addEventListener("click", (e) => {
    if (!completedDrag || e.detail === 0 || (e.sourceCapabilities && !e.sourceCapabilities.firesTouchEvents)) return;
    const { startX, startY, endX, endY, time } = completedDrag;
    completedDrag = null;
    if (Date.now() - time < 1000 &&
        (Math.hypot(e.clientX - startX, e.clientY - startY) <= 25 ||
         Math.hypot(e.clientX - endX, e.clientY - endY) <= 25)) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, true);
  nav.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const handler = navActions[btn.dataset.action];
      if (handler) handler(e, btn);
    });
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".nav-dropdown")) closeDropdowns(); });

  const eye = document.querySelector("#hal");
  if (eye) {
    const hasMouse = window.matchMedia("(hover: hover) and (pointer: fine)");
    document.addEventListener("mousemove", (e) => {
      if (hasMouse.matches) lookAt(eye, e.clientX, e.clientY);
    });
    document.addEventListener("mousedown", (e) => {
      if (!hasMouse.matches) lookAt(eye, e.clientX, e.clientY);
    });
    document.addEventListener("touchstart", (e) => {
      completedDrag = null;
      if (touchDrag) return;
      const touch = e.changedTouches[0];
      if (touch) {
        touchDrag = { id: touch.identifier, startX: touch.clientX, startY: touch.clientY, moved: false };
        lookAt(eye, touch.clientX, touch.clientY);
      }
    }, { passive: true });
    document.addEventListener("touchmove", (e) => {
      if (!touchDrag) return;
      const touch = Array.from(e.changedTouches).find((t) => t.identifier === touchDrag.id);
      if (!touch) return;
      if (Math.hypot(touch.clientX - touchDrag.startX, touch.clientY - touchDrag.startY) > 5) touchDrag.moved = true;
      lookAt(eye, touch.clientX, touch.clientY);
    }, { passive: true });
    document.addEventListener("touchend", (e) => {
      if (!touchDrag) return;
      const touch = Array.from(e.changedTouches).find((t) => t.identifier === touchDrag.id);
      if (!touch) return;
      if (touchDrag.moved || Math.hypot(touch.clientX - touchDrag.startX, touch.clientY - touchDrag.startY) > 5) {
        completedDrag = { ...touchDrag, endX: touch.clientX, endY: touch.clientY, time: Date.now() };
      }
      touchDrag = null;
    }, { passive: true });
    document.addEventListener("touchcancel", (e) => {
      if (touchDrag && Array.from(e.changedTouches).some((t) => t.identifier === touchDrag.id)) touchDrag = null;
    }, { passive: true });
    window.addEventListener("resize", () => {
      if (lastEyePoint) lookAt(eye, lastEyePoint.x, lastEyePoint.y);
    });
  }

  // opacity: 1 øverst og ved scroll op; falder gradvist ved scroll ned
  let lastY = window.scrollY, op = 1;
  const fade = 400; // px scroll for fuld fade
  window.addEventListener("scroll", () => {
    const y = window.scrollY, dy = y - lastY;
    lastY = y;
    if (y <= 0 || dy < 0) op = 1;
    else op = Math.max(0, op - dy / fade);
    nav.style.opacity = op;
    nav.classList.toggle("faded", op < 0.2);
  }, { passive: true });

  const root = document.getElementById("content");
  const truncs = root.querySelectorAll('#content .truncate');
  const HOLD_MS = 250, DRAG_PX = 5;
  let down = null;
  // hold og drag skal ikke aktiverer .truncable
  document.addEventListener("mousedown", (e) => { 
    down = { x: e.clientX, y: e.clientY, t: Date.now() }; 
  });
  const isHoldOrDrag = (e) => {
    if (down && (Date.now() - down.t > HOLD_MS || Math.hypot(e.clientX - down.x, e.clientY - down.y) > DRAG_PX)) return true;
    const sel = window.getSelection();
    return !!sel && !sel.isCollapsed; // tekstmarkering efter træk
  };
  truncs.forEach((h) => h.addEventListener("click", (e) => {
    if (isHoldOrDrag(e)) return;
    h.classList.toggle("expanded", !h.classList.contains("expanded"));
  }));


  // fold teksten ud, hvis man lander på et #anker på en h6
  const openHash = () => {
    const id = decodeURIComponent(location.hash.slice(1));
    const target = id && document.getElementById(id);
    if (target && target.tagName === "H5") target.classList.add("expanded");
  };
  window.addEventListener("hashchange", openHash);
  openHash();


  
  
});