browser.tabs.query({active: true, currentWindow: true}, function(tabs) {
    console.log('success');
});