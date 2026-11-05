
const pdfInput = document.getElementById("pdfInput");
const dropZone = document.getElementById("dropZone");

pdfInput.addEventListener("change", function () {
    handleFiles(this.files);
});

function handleFiles(files) {
    Array.from(files).forEach(file => {
        if (file.type !== "application/pdf") {
            alert("Please upload a PDF file.");
            return;
        }

        if (file.size > 20 * 1024 * 1024) {
            alert("File must be smaller than 20 MB.");
            return;
        }

        const list = document.getElementById("documentList");

        if (list.querySelector(".empty-state")) {
            list.innerHTML = "";
        }

        const item = document.createElement("div");
        item.className = "document-item";

        const name = document.createElement("span");
        name.textContent = "📄 " + file.name;

        const status = document.createElement("span");
        status.textContent = "Uploaded locally";
        status.style.color = "#32aa82";

        item.append(name, status);
        list.appendChild(item);

        document.getElementById("fileStatus").textContent =
            "Selected: " + file.name;
    });
}

dropZone.addEventListener("dragover", event => {
    event.preventDefault();
    dropZone.style.background = "#f5f2ff";
});

dropZone.addEventListener("dragleave", () => {
    dropZone.style.background = "white";
});

dropZone.addEventListener("drop", event => {
    event.preventDefault();
    dropZone.style.background = "white";
    handleFiles(event.dataTransfer.files);
});

function newChat() {
    alert("New chat interface will open here.");
}

function openFeature(feature) {
    alert("The " + feature + " module will be developed next.");
}
