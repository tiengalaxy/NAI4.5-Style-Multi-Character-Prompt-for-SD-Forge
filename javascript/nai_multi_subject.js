(function () {
    function init() {
        var interval = setInterval(function () {
            var tabsContainer = document.querySelector("#nai_char_section");
            if (!tabsContainer) return;

            var accordions = tabsContainer.querySelectorAll("[id^='nai_char_acc_']");
            if (accordions.length === 0) return;

            clearInterval(interval);

            var borderColors = [
                "4px solid #e74c3c",
                "4px solid #3498db",
                "4px solid #2ecc71",
                "4px solid #f39c12"
            ];

            accordions.forEach(function (acc, index) {
                if (index < borderColors.length) {
                    var inner = acc.querySelector(".label-wrap") || acc;
                    inner.style.borderLeft = borderColors[index];
                    inner.style.borderRadius = "0 8px 8px 0";
                    inner.style.paddingLeft = "12px";
                }
            });

            console.log("[NAI Multi-Subject] Character color borders applied.");
        }, 1000);

        setTimeout(function () {
            clearInterval(interval);
        }, 15000);

        setupPreviewPolling();
    }

    var naiPreviewInterval = null;

    function setupPreviewPolling() {
        var genBtn = document.querySelector("#nai_gen_btn");
        if (!genBtn) {
            setTimeout(setupPreviewPolling, 2000);
            return;
        }

        genBtn.addEventListener("click", function () {
            startPreviewPolling();
        });
    }

    function startPreviewPolling() {
        stopPreviewPolling();
        var container = document.querySelector("#nai_preview_container");
        if (container) container.style.display = "block";

        naiPreviewInterval = setInterval(function () {
            fetch("/sdapi/v1/progress")
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    var img = document.querySelector("#nai_preview_img");
                    var progress = document.querySelector("#nai_preview_progress");

                    if (data.current_image && img) {
                        img.src = "data:image/png;base64," + data.current_image;
                    }

                    if (progress) {
                        var pct = Math.round(data.progress * 100);
                        var eta = data.eta_relative ? Math.round(data.eta_relative) : "?";
                        progress.textContent = "Progress: " + pct + "% | ETA: " + eta + "s";
                    }
                })
                .catch(function () {});
        }, 1000);
    }

    function stopPreviewPolling() {
        if (naiPreviewInterval) {
            clearInterval(naiPreviewInterval);
            naiPreviewInterval = null;
        }
        var container = document.querySelector("#nai_preview_container");
        if (container) container.style.display = "none";
    }

    var galleryObserver = new MutationObserver(function () {
        var gallery = document.querySelector("#nai_gallery");
        if (gallery && gallery.children.length > 0) {
            stopPreviewPolling();
        }
    });

    setTimeout(function () {
        var gallery = document.querySelector("#nai_gallery");
        if (gallery) {
            galleryObserver.observe(gallery, { childList: true, subtree: true });
        }
    }, 3000);

    if (typeof onUiLoaded === "function") {
        onUiLoaded(init);
    } else {
        document.addEventListener("DOMContentLoaded", function () {
            setTimeout(init, 1500);
        });
    }
})();
