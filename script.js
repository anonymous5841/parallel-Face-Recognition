const cardsContainer = document.getElementById('cardsContainer');
const cards = document.querySelectorAll('.card');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

let currentIndex = 0;

/* -------------------------
   URL-based initial selection
------------------------- */
const params = new URLSearchParams(window.location.search);
const active = params.get("active");

// Map string to index
if (active === "recognize") currentIndex = 1;
if (active === "dataset") currentIndex = 2;

// Update container transform based on index
function moveCarousel() {
    const cardWidth = cards[0].offsetWidth + 30; // adjust gap if needed
    cardsContainer.style.transform = `translateX(${-currentIndex * cardWidth}px)`;
}

/* -------------------------
   Update carousel cards
------------------------- */
document.addEventListener("DOMContentLoaded", () => {
    const cardsContainer = document.getElementById('cardsContainer');
    const cards = document.querySelectorAll('.card');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');

    let currentIndex = 0;

    // URL-based initial selection
    const params = new URLSearchParams(window.location.search);
    const active = params.get("active");

    if (active === "recognize") currentIndex = 1;
    if (active === "dataset") currentIndex = 2;

    function updateCarousel() {
        cards.forEach((card, idx) => {
            const video = card.querySelector("video");
            if (idx === currentIndex) {
                card.classList.add("selected");
                if (video) {
                    video.currentTime = 0;
                    video.play();
                }
            } else {
                card.classList.remove("selected");
                if (video) video.pause();
            }
        });

        // Let your existing carousel CSS handle positioning; do not manually translate
    }

    prevBtn.addEventListener('click', () => {
        if (currentIndex > 0) {
            currentIndex--;
            updateCarousel();
        }
    });

    nextBtn.addEventListener('click', () => {
        if (currentIndex < cards.length - 1) {
            currentIndex++;
            updateCarousel();
        }
    });

    // Initialize carousel after DOM ready
    updateCarousel();
});
