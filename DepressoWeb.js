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

    const values = [envsat,achievesat,finstress,insomnia,anxiety,deprived,abused,cheated,threatened,suicidal,inferiority,reccon,recloss];
    return values
}

function getModel(){
	const radio = document.getElementById("radio-right");
	const option = radio.querySelector("#model").value;
	return option;
}

function displayResults(result){
    let result_text
    if (result > 0.5){
        result_text = 'High probability of depression';
    } else{
        result_text = 'Low probability of depression';
    }
    let textEl = document.getElementById("result");
    textEl = textEl.querySelector("p");
    if (Number.isInteger(result)){
        textEl.innerHTML= result_text;
    } else{
        textEl.innerHTML= result_text + ": " + (result * 100) + "%";
    }
}

async function submitResults(){
	values = getValues();
	model = getModel();

	const obj = {"values" : values, "model" : model};
	const json_str = JSON.stringify(obj);

    const resp = await fetch('http://127.0.0.1:5000/'+json_str)
    .then(response => response.json())
    .then(json => {return json;})
    displayResults(resp['prediction']);
}