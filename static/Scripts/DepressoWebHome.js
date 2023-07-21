const url = 'http://127.0.0.1:5000/home/'

window.onscroll = function(){
    getScrollPercent();
};
function getScrollPercent(){
    const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    let scrollPercent = winScroll / height;
    scrollPercent = scrollPercent * 5; // reached 100% in 20% of the webpage
    if (scrollPercent > 1){
        scrollPercent = 1;
    }
    //navbarShrink(scrollPercent);
    navbarTextOpacityAnimation(scrollPercent);
}

/*
function navbarShrinkAnimation(scrollPercent) {
    const navBar = document.querySelector("nav");
    const navHeight = 80; // 80px
    let newNavHeight = navHeight - (scrollPercent * navHeight * 0.3); // navbar shrinks to 70% (1-0.3) of its size
    newNavHeight = newNavHeight+"px";
    navBar.style.height = newNavHeight;
}
*/

function navbarTextOpacityAnimation(scrollPercent) {
    scrollPercent = scrollPercent * 5; // reached 100% in 20% of the webpage
    if (scrollPercent > 1){
        scrollPercent = 1;
    }
    const navText = document.getElementById("nav-text");
    navText.style.opacity = 1 - scrollPercent; // navbar text opacity
}