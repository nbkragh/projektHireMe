document.addEventListener("DOMContentLoaded", () => {
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