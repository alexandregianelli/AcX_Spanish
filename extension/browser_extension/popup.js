url_server = "http://0.0.0.0:5000/article_url/";

function addLine(tableID, acro, exp) {
    var refTable = document.getElementById(tableID);
    var newLine = refTable.insertRow(-1);
    var newCellAcro = newLine.insertCell(0);
    var newCellExp = newLine.insertCell(1);
    var newTextAcro = document.createTextNode(acro);
    var newTextExp = document.createTextNode(exp);
    newCellAcro.appendChild(newTextAcro);
    newCellExp.appendChild(newTextExp);
    }


function addLines(text){
    var keys = {}
    for (var key in text){
        var value = text[key];
        addLine('display', key, value);
        keys[key] = value;
    } 
    browser.runtime.sendMessage({  //send a message to the background script
        from: "popup",
        keys:keys
    });
}


browser.tabs.query({'active': true, 'windowId': browser.windows.WINDOW_ID_CURRENT},
    function(tabs){
        var bg = chrome.extension.getBackgroundPage();
        var res = bg.results;
        var active_url = String(tabs[0].url);
        while (active_url.includes("/")){
            active_url = active_url.replace("/", "@@");
        }
        console.log("DICT :", res);
        if (active_url in res){
            addLines(res[active_url]);
        }else{
            let request = new XMLHttpRequest();
            request.open("GET",url_server+active_url);
            request.send();
            request.onload = () =>{
                var request_JSON = JSON.parse(request.responseText);
                addLines(request_JSON);
                bg.saveRes(active_url, request_JSON); 
            }
        }
    }
 );

