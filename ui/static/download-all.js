// "Download all": build the ZIP in the visitor's browser.
//
// The server only returns a list of photo URLs; the browser fetches each photo
// directly (from R2 in production) and packs them with client-zip, so photos
// never stream through the web server. Photos are stored uncompressed, as
// JPEGs do not shrink further.
//
// Where the File System Access API exists (desktop Chrome/Edge) the archive is
// streamed straight to disk; elsewhere it is assembled as a Blob and saved.
// Browsers that cannot build a ZIP download the photos one by one instead
// (download-each.js). The button stays hidden when JavaScript is off; each
// photo keeps its own Download link.

import { downloadZip } from "./vendor/client-zip-2.4.5.js";

const button = document.querySelector("[data-download-all]");

const canZip =
  "ReadableStream" in window &&
  "fetch" in window &&
  "Response" in window &&
  typeof BigInt === "function" &&
  typeof Response.prototype.blob === "function";

if (button && !canZip) {
  button.querySelector("[data-label]").textContent = "Download all";
  button.hidden = false;
  button.addEventListener("click", (event) => {
    event.preventDefault();
    window.rpfDownloadEach();
  });
} else if (button) {
  button.hidden = false;
  const label = button.querySelector("[data-label]");
  const status = document.querySelector("[data-download-status]");
  const idleText = label.textContent;
  let busy = false;

  const setStatus = (text, isError = false) => {
    status.textContent = text;
    status.hidden = !text;
    status.classList.toggle("error", isError);
  };

  button.addEventListener("click", async (event) => {
    event.preventDefault();
    if (busy) return;
    busy = true;
    button.disabled = true;
    setStatus("");

    // The save dialog must open directly from the click, before any await.
    let fileHandle = null;
    const pickerPromise =
      "showSaveFilePicker" in window
        ? window
            .showSaveFilePicker({
              suggestedName: button.dataset.archive,
              types: [{ description: "ZIP archive", accept: { "application/zip": [".zip"] } }],
            })
            .catch((error) => (error.name === "AbortError" ? "cancelled" : null))
        : Promise.resolve(null);

    try {
      fileHandle = await pickerPromise;
      if (fileHandle === "cancelled") return;

      label.textContent = "Preparing…";
      const response = await fetch(button.dataset.manifest, { credentials: "same-origin" });
      if (!response.ok) throw new Error(`Photo list failed (${response.status})`);
      const manifest = await response.json();
      const total = manifest.files.length;
      let done = 0;

      async function* photos() {
        for (const file of manifest.files) {
          label.textContent = `Preparing ${done + 1} / ${total}…`;
          const photo = await fetch(file.url);
          if (!photo.ok) throw new Error(`Could not download ${file.name}`);
          yield { name: file.name, input: photo };
          done += 1;
        }
      }

      const zip = downloadZip(photos());
      if (fileHandle) {
        const writable = await fileHandle.createWritable();
        await zip.body.pipeTo(writable);
      } else {
        const blob = await zip.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = manifest.archive_name;
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      }
      setStatus(`Saved ${total} photo${total === 1 ? "" : "s"}.`);
    } catch (error) {
      console.error(error);
      setStatus("The download did not finish. Please try again.", true);
    } finally {
      label.textContent = idleText;
      button.disabled = false;
      busy = false;
    }
  });
}
