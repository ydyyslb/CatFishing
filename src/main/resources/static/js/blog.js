document.addEventListener("DOMContentLoaded", () => {
    navigate('home');
});


function navigate(section) {
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.classList.remove('active');
    });
    
    const selectedSection = document.getElementById(section);
    if (selectedSection) {
        selectedSection.classList.add('active');
    }

    const navItems = document.querySelectorAll('.sidebar ul li');
    navItems.forEach(item => {
        item.classList.remove('active');
    });
    const activeNav = document.getElementById(`${section}-nav`);
    if (activeNav) {
        activeNav.classList.add('active');
    }
}

function updateSidebar(activeSectionId) {
    const sidebarItems = document.querySelectorAll('.sidebar ul li');
    sidebarItems.forEach(item => {
        item.classList.remove('active');
    });

    const activeSidebarItem = document.getElementById(`${activeSectionId}-nav`);
    if (activeSidebarItem) {
        activeSidebarItem.classList.add('active');
    }
}

let currentScrollPosition = 0;

function scrollCarousel(direction) {
    const carouselContent = document.querySelector('.carousel-content');
    const itemWidth = carouselContent.querySelector('.carousel-item').clientWidth;
    const visibleItems = 4;
    const totalItems = carouselContent.children.length;
    const maxScrollPosition = itemWidth * (totalItems - visibleItems);

    if (direction === 'left') {
        currentScrollPosition -= itemWidth;
        if (currentScrollPosition < 0) {
            currentScrollPosition = 0;
        }
    } else if (direction === 'right') {
        currentScrollPosition += itemWidth;
        if (currentScrollPosition > maxScrollPosition) {
            currentScrollPosition = maxScrollPosition;
        }
    }

    carouselContent.style.transform = `translateX(-${currentScrollPosition}px)`;
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const content = document.querySelector('.content');

    if (sidebar.style.transform === 'translateX(-200px)') {
        sidebar.style.transform = 'translateX(0)';
        content.classList.remove('centered');
        content.classList.add('shifted');
    } else {
        sidebar.style.transform = 'translateX(-200px)';
        content.classList.remove('shifted');
        content.classList.add('centered');
    }
}

