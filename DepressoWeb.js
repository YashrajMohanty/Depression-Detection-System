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
	let option = radio.querySelector("#model").value;
    option = Number(option);
	return option;
}

function displayResults(result, model){
    changeResultsDivColor(result);
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
    navScrollAnimation();
};
function navScrollAnimation(){
    const navText = document.getElementById("nav-text");
    const navBar = document.querySelector("nav");
    const navHeight = 80; // 80px
    const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    let scrollPercent = winScroll / height;
    scrollPercent = scrollPercent * 5; // reached 100% in 20% of the webpage
    if (scrollPercent > 1){
        scrollPercent = 1;
    }
    navText.style.opacity = 1 - scrollPercent; // navbar text opacity
    let newNavHeight = navHeight - (scrollPercent * navHeight * 0.3); // navbar shrinks to 70% (1-0.3) of its size
    newNavHeight = newNavHeight+"px";
    navBar.style.height = newNavHeight;
}

function changeResultsDivColor(result){
    const resultDiv = document.getElementById("result");
    const currentColor = resultDiv.getAttribute('style','background');

    let newColor;
    if (result > 0.5){
        newColor = '#ffcfcf'; //red
    }
    else {
        newColor = '#c6ffb5'; //green
    }
    if (result === -1){
        newColor = '#fcd9a7'; //amber
    }

    keyFrames = [{background: currentColor},{background: newColor},{background: currentColor}];
    animTiming = {duration: 1000, iterations: 1};
    resultDiv.animate(keyFrames, animTiming);
}

async function submitResults(){
	values = getValues();
	model = getModel();
	const obj = {"values" : values, "model" : model};
	const json_str = JSON.stringify(obj);
    const url = ''

    try{

        const resp = await fetch(url + json_str, {signal: AbortSignal.timeout(2000)})
        .then(response => response.json())
        .then(json => {return json});
        displayResults(resp['prediction'], model);

    } catch(err){
        displayResults(-1, model);
    }

}