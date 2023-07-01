const url = 'http://127.0.0.1:5000/depdet/';


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


function getClientLocation(){
    navigator.permissions.query({name:"geolocation"})
    .then((result) => {
        if (result.state === "granted"){
            console.log(result.state);
            displayMap();
        }
        else if (result.state === "prompt"){
            console.log(result.state);
            displayMap();
        }
        else if (result.state === "denied"){
            console.log(result.state);
        }
    })
}


function displayMap(){
    const mapFrame = document.querySelector("div.map iframe");
    if (mapFrame.getAttribute("src") === ""){
        navigator.geolocation.getCurrentPosition((position) => {
            console.log("Latitude: "+position.coords.latitude+", Longitude: "+position.coords.longitude); 
            const mapUrl = "http://www.google.com/maps?q=psychiatrist+near+me/"+position.coords.latitude+","+position.coords.longitude+"&z=13&output=embed";
            mapFrame.setAttribute("src",mapUrl);
        });
    }
    mapFrame.hidden = !mapFrame.hidden;
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


function getValues(){
    const radio = document.querySelector("div.radio-parent");
    const nameList = ['envsat','achievesat','finstress','insomnia','anxiety','deprived','abused',
                    'cheated','threatened','suicidal','inferiority','reccon','recloss'];
    let values = [];

    nameList.forEach((name) => {
        values.push(radio.querySelector("input[name="+ name +"]:checked".value));
    });

    values = values.map(x => Number(x));
    console.log(values);
    return values;
}


function getModel(){
	const radio = document.querySelector("div.radio-parent");
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
    const newColor = 'hsla(43, 100%, 70%, 0.5)'
    keyFrames = [{background: currentColor},{background: newColor},{background: currentColor}];
    animTiming = {duration: 1000, iterations: 1};
    optionsBar.animate(keyFrames, animTiming);
}