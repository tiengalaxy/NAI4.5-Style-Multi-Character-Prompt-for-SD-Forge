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
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(init);
    } else {
        document.addEventListener("DOMContentLoaded", function () {
            setTimeout(init, 1500);
        });
    }
})();
