/**
 * Laura Image Studio - Model Links Web Extension
 *
 * Intercepts the LauraModelLinks node to render [LINK:filename|url] markers
 * as clickable hyperlinks, and displays the ASCII directory tree with proper
 * formatting. This gives users one-click access to HuggingFace model downloads.
 *
 * Pattern: ComfyUI web extension using app.registerExtension()
 */

import { app } from "../../scripts/app.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Parse a text block and replace [LINK:filename|url] markers with <a> elements.
 * Returns an HTML string safe for innerHTML (all non-link text is escaped).
 */
function renderLinksAsHTML(text) {
    if (!text) return "";

    const linkRegex = /\[LINK:([^|]+)\|([^\]]+)\]/g;
    const parts = [];
    let lastIndex = 0;

    let match;
    while ((match = linkRegex.exec(text)) !== null) {
        // Escape text before the link
        if (match.index > lastIndex) {
            parts.push(escapeHTML(text.slice(lastIndex, match.index)));
        }
        const filename = escapeHTML(match[1]);
        const url = escapeHTML(match[2]);
        parts.push(
            '<a href="' + url + '" target="_blank" rel="noopener noreferrer" ' +
            'class="laura-model-link" title="Download ' + filename + '">' + filename + '</a>'
        );
        lastIndex = match.index + match[0].length;
    }

    // Remaining text after last link
    if (lastIndex < text.length) {
        parts.push(escapeHTML(text.slice(lastIndex)));
    }

    return parts.join("");
}

function escapeHTML(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// Styles (injected once)
// ---------------------------------------------------------------------------

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;

    const style = document.createElement("style");
    style.textContent = [
        "/* Laura Model Links - Node Display */",
        ".laura-model-links-container {",
        "    font-family: 'Courier New', monospace;",
        "    font-size: 11px;",
        "    line-height: 1.5;",
        "    color: #e0e0e0;",
        "    background: #1a1a2e;",
        "    border: 1px solid #333;",
        "    border-radius: 6px;",
        "    padding: 10px;",
        "    max-height: 400px;",
        "    overflow-y: auto;",
        "    white-space: pre-wrap;",
        "    word-break: break-all;",
        "}",
        ".laura-model-links-container .laura-model-link {",
        "    color: #ffd700;",
        "    text-decoration: underline;",
        "    cursor: pointer;",
        "    font-weight: bold;",
        "    transition: color 0.15s ease;",
        "}",
        ".laura-model-links-container .laura-model-link:hover {",
        "    color: #87ceeb;",
        "    text-decoration: underline;",
        "}",
        ".laura-model-links-container .laura-section-header {",
        "    color: #87ceeb;",
        "    font-weight: bold;",
        "}",
        ".laura-model-links-container .laura-separator {",
        "    color: #555;",
        "}",
        ".laura-directory-tree {",
        "    font-family: 'Courier New', monospace;",
        "    font-size: 11px;",
        "    line-height: 1.6;",
        "    color: #c0c0c0;",
        "    background: #16213e;",
        "    border: 1px solid #333;",
        "    border-radius: 6px;",
        "    padding: 10px;",
        "    white-space: pre;",
        "    overflow-x: auto;",
        "    margin-top: 6px;",
        "}",
        ".laura-directory-tree .laura-tree-folder {",
        "    color: #ffd700;",
        "    font-weight: bold;",
        "}",
        ".laura-directory-tree .laura-tree-file {",
        "    color: #90ee90;",
        "}",
        ".laura-links-tabs {",
        "    display: flex;",
        "    gap: 4px;",
        "    margin-bottom: 6px;",
        "}",
        ".laura-links-tab {",
        "    padding: 3px 10px;",
        "    background: #2a2a4e;",
        "    border: 1px solid #444;",
        "    border-radius: 4px 4px 0 0;",
        "    color: #aaa;",
        "    cursor: pointer;",
        "    font-size: 10px;",
        "    font-family: sans-serif;",
        "}",
        ".laura-links-tab.active {",
        "    background: #1a1a2e;",
        "    color: #ffd700;",
        "    border-bottom-color: #1a1a2e;",
        "}",
    ].join("\n");
    document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// Render the directory tree with syntax highlighting
// ---------------------------------------------------------------------------

function renderDirectoryTree(text) {
    if (!text) return "";
    // Highlight folder names (ending with /) and file names (.safetensors etc.)
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(
            /(\S+\/)/g,
            '<span class="laura-tree-folder">$1</span>'
        )
        .replace(
            /(\S+\.safetensors|\S+\.ckpt|\S+\.pt|\S+\.pth|\S+\.bin)/g,
            '<span class="laura-tree-file">$1</span>'
        );
}

// ---------------------------------------------------------------------------
// Extension Registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "laura.studio.model_links",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (!nodeData || nodeData.name !== "LauraModelLinks") {
            return;
        }

        injectStyles();

        // Store original onExecuted
        const origOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (output) {
            // Call original if present
            if (origOnExecuted) {
                origOnExecuted.call(this, output);
            }

            // Extract outputs
            const modelLinks = (output && output.model_links && output.model_links[0]) || "";
            const directoryTree = (output && output.directory_tree && output.directory_tree[0]) || "";

            // Find or create display container
            let container = this._lauraLinksContainer;
            if (!container) {
                // Create the DOM container and attach it as a custom widget
                container = document.createElement("div");
                container.style.width = "100%";
                this._lauraLinksContainer = container;

                // Create tabs
                const tabs = document.createElement("div");
                tabs.className = "laura-links-tabs";

                const linksTab = document.createElement("div");
                linksTab.className = "laura-links-tab active";
                linksTab.textContent = "Download Links";

                const treeTab = document.createElement("div");
                treeTab.className = "laura-links-tab";
                treeTab.textContent = "Directory Tree";

                tabs.appendChild(linksTab);
                tabs.appendChild(treeTab);

                // Create content areas
                const linksContent = document.createElement("div");
                linksContent.className = "laura-model-links-container";

                const treeContent = document.createElement("div");
                treeContent.className = "laura-directory-tree";
                treeContent.style.display = "none";

                // Tab switching
                linksTab.addEventListener("click", function () {
                    linksTab.classList.add("active");
                    treeTab.classList.remove("active");
                    linksContent.style.display = "block";
                    treeContent.style.display = "none";
                });
                treeTab.addEventListener("click", function () {
                    treeTab.classList.add("active");
                    linksTab.classList.remove("active");
                    treeContent.style.display = "block";
                    linksContent.style.display = "none";
                });

                container.appendChild(tabs);
                container.appendChild(linksContent);
                container.appendChild(treeContent);

                this._lauraLinksDiv = linksContent;
                this._lauraTreeDiv = treeContent;

                // Add as a DOM widget to the node
                var widget = this.addDOMWidget(
                    "laura_links_display",
                    "customtext",
                    container,
                    {
                        serialize: false,
                        hideOnZoom: false,
                    }
                );
                widget.computeSize = function () {
                    return [340, 300];
                };
            }

            // Render content
            if (this._lauraLinksDiv) {
                this._lauraLinksDiv.innerHTML = renderLinksAsHTML(modelLinks);
            }
            if (this._lauraTreeDiv) {
                this._lauraTreeDiv.innerHTML = renderDirectoryTree(directoryTree);
            }

            // Resize node to fit
            this.setSize([
                Math.max(this.size[0], 360),
                Math.max(this.size[1], 380),
            ]);
        };
    },
});
