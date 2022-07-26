var url = window.location.href.toString();
var options = {
  "accuracy": {
    "value": "exactly",
    "limiters": ["'",'"',":","-","[","]",",",".",")","("]},
  "caseSensitive": true
}

chrome.runtime.sendMessage({from:"content"}); //first, tell the background page that this is the tab that wants to receive the messages.

function changePage(acronyms){
    var context = document.querySelectorAll("body");
    var instance = new Mark(context);
    for (var key in acronyms){
      options.className = key;
      instance.mark(key, options);
      $('.'+key).tooltip({title: acronyms[key]});
    }
}

chrome.runtime.onMessage.addListener(function(msg) {
  if (msg.from == "background") {
    changePage(msg.keys);
  }
});
