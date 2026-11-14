// "Download all" fallback: download the photos one by one.
//
// Used when the browser cannot build a ZIP itself. Written as a classic ES5
// script on purpose: the oldest browsers never run module scripts, so this is
// the part that must work everywhere. download-all.js (a module) reuses it via
// window.rpfDownloadEach when a modern browser lacks the streaming features
// the ZIP builder needs.
(function () {
  "use strict";

  var button = document.querySelector("[data-download-all]");
  if (!button) return;
  var label = button.querySelector("[data-label]");
  var status = document.querySelector("[data-download-status]");

  function setStatus(text, isError) {
    status.textContent = text;
    status.hidden = !text;
    if (isError) status.className = "download-status error";
    else status.className = "download-status";
  }

  function downloadEach() {
    var request = new XMLHttpRequest();
    request.open("GET", button.getAttribute("data-manifest"));
    request.onload = function () {
      if (request.status !== 200) {
        setStatus("The download did not start. Please try again.", true);
        return;
      }
      var files = JSON.parse(request.responseText).files;
      // Space the downloads out so browsers do not drop any of them.
      for (var i = 0; i < files.length; i++) {
        (function (file, delay) {
          setTimeout(function () {
            var link = document.createElement("a");
            link.href = file.download_url;
            link.setAttribute("download", file.name);
            link.style.display = "none";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
          }, delay);
        })(files[i], i * 800);
      }
      setStatus(
        "Downloading " + files.length + " photos one by one. " +
        "If your browser asks, allow multiple downloads."
      );
    };
    request.onerror = function () {
      setStatus("The download did not start. Please try again.", true);
    };
    request.send();
  }

  window.rpfDownloadEach = downloadEach;

  // Browsers without module support never run download-all.js: handle the
  // button here instead.
  var supportsModules = "noModule" in document.createElement("script");
  if (!supportsModules) {
    label.textContent = "Download all";
    button.hidden = false;
    button.onclick = function (event) {
      if (event && event.preventDefault) event.preventDefault();
      downloadEach();
    };
  }
})();
