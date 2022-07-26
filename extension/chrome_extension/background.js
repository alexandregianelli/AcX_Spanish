console.log("AcroDisam Start")

var results = {};

var contentTabId;

chrome.runtime.onMessage.addListener(function(msg,sender) {
  if (msg.from == "content") {  //get content scripts tab id
    contentTabId = sender.tab.id;
  }
  if (msg.from == "popup" && contentTabId) {  //got message from popup
    chrome.tabs.sendMessage(contentTabId, {  //send it to content script
      from: "background",
      keys: msg.keys
    });
  }
});