const url = 'http://127.0.0.1:5000/depdet/'

function getValues(){
    let radio = document.getElementById("radio-left");
    const envsat = radio.querySelector("input[name='envsat']:checked").value;
    const achievesat = radio.querySelector("input[name='achievesat']:checked").value;
    const finstress = radio.querySelector("input[name='finstress']:checked").value;
    const insomnia = radio.querySelector("input[name='insomnia']:checked").value;
    const anxiety = radio.querySelector("input[name='anxiety']:checked").value;
    const deprived = radio.querySelector("input[name='deprived']:checked").value;
    const abused = radio.querySelector("input[name='abused']:checked").value;
    radio = document.getElementById("radio-right");
    const cheated = radio.querySelector("input[name='cheated']:checked").value;
    const threatened = radio.querySelector("input[name='threatened']:checked").value;
    const suicidal = radio.querySelector("input[name='suicidal']:checked").value;
    const inferiority = radio.querySelector("input[name='inferiority']:checked").value;
    const reccon = radio.querySelector("input[name='reccon']:checked").value;
    const recloss = radio.querySelector("input[name='recloss']:checked").value;

    let values = [envsat,achievesat,finstress,insomnia,anxiety,deprived,abused,cheated,threatened,suicidal,inferiority,reccon,recloss];
    values = values.map(x => Number(x));
    return values;
}

function getModel(){
	const radio = document.getElementById("radio-right");
	let option = radio.querySelector("#model-select").value;
    option = Number(option);
	return option;
}

function displayResults(result, model){
    changeResultTextColor(result);
    let result_text
    
    if (result > 0.5){
        result_text = 'High probability of depression';
    } else{
        result_text = 'Low probability of depression';
    }

    if (model == 5|model == 7){
        result_text = result_text + ": " + (result * 100) + "%";
    }

    if (result === -1){
        result_text = "Connection timed out!";
    }

    let textEl = document.getElementById("result");
    textEl = textEl.querySelector("p");
    textEl.innerHTML= result_text;
}

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

function navbarTextOpacityAnimation(scrollPercent) {
    const navText = document.getElementById("nav-text");
    navText.style.opacity = 1 - scrollPercent; // navbar text opacity
}

function changeResultTextColor(result){
    const resultDiv = document.getElementById("result");
    const currentColor = resultDiv.getAttribute('style','color');

    let newColor;
    if (result > 0.5){
        newColor = 'hsla(0, 100%, 50%, 0.8)'; //red
    }
    else {
        newColor = 'hsla(113, 98%, 49%, 0.8)'; //green
    }
    if (result === -1){
        newColor = 'hsla(38, 93%, 72%, 0.8)'; //amber
    }

    keyFrames = [{color: currentColor},{color: newColor},{color: currentColor}];
    animTiming = {duration: 1000, iterations: 1};
    resultDiv.animate(keyFrames, animTiming);
}

function flashOptionsBar(){
    const optionsBar = document.getElementById('model-select');
    const currentColor = optionsBar.getAttribute('style','background');
    const newColor = 'hsla(43, 100%, 70%, 0.3)'
    keyFrames = [{background: currentColor},{background: newColor},{background: currentColor}];
    animTiming = {duration: 1000, iterations: 1};
    optionsBar.animate(keyFrames, animTiming);
}

async function submitResults(){
	values = getValues();
	model = getModel();

    if (model === -1){
        flashOptionsBar();
        return;
    }

	const obj = {"values" : values, "model" : model};
	const json_str = JSON.stringify(obj);

    try{
        const resp = await fetch(url + json_str, {signal: AbortSignal.timeout(2000)})
        .then(response => response.json())
        .then(json => {return json});
        displayResults(resp['prediction'], model);

    } catch(err){
        displayResults(-1, model);
    }
}