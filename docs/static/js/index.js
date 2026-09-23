// Trimmed from the Academic Project Page Template's static/js/index.js: only the
// scroll-to-top button is used on this page (no carousels, sliders, BibTeX or
// "More Works" dropdown), so jQuery and the carousel/slider scripts aren't needed.

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function () {
  const scrollButton = document.querySelector('.scroll-to-top');
  if (window.pageYOffset > 300) {
    scrollButton.classList.add('visible');
  } else {
    scrollButton.classList.remove('visible');
  }
});
