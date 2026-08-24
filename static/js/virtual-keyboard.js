/**
 * virtual-keyboard.js - Integrated Touch On-Screen Keyboard for Kiosk
 * Designed for 100% Offline Raspberry Pi Touchscreens & Chromium Kiosk
 */

(function () {
    'use strict';

    // Keyboard State
    let activeInput = null;
    let isShift = false;
    let isCapsLock = false;
    let currentLayout = 'qwerty'; // 'qwerty' | 'symbols' | 'numpad'
    let isVisible = false;

    // Layout Definitions
    const LAYOUT_QWERTY = [
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
        ['shift', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'backspace'],
        ['?123', 'tab', 'space', 'clear', 'enter']
    ];

    const LAYOUT_SYMBOLS = [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['@', '#', '$', '%', '&', '-', '+', '(', ')', '/'],
        ['*', '"', "'", ':', ';', '!', '?', ',', '.', 'backspace'],
        ['ABC', 'tab', 'space', 'clear', 'enter']
    ];

    const LAYOUT_NUMPAD = [
        ['1', '2', '3'],
        ['4', '5', '6'],
        ['7', '8', '9'],
        ['.', '0', 'backspace'],
        ['ABC', 'clear', 'tab', 'enter']
    ];

    // Build DOM Elements
    let wrapper = null;
    let previewEl = null;
    let fieldInfoEl = null;
    let bodyEl = null;
    let floatToggleBtn = null;

    function initKeyboard() {
        if (document.getElementById('vk-container')) return;

        // Container
        wrapper = document.createElement('div');
        wrapper.id = 'vk-container';
        wrapper.className = 'vk-wrapper';

        // Header
        const header = document.createElement('div');
        header.className = 'vk-header';

        fieldInfoEl = document.createElement('div');
        fieldInfoEl.className = 'vk-field-info';
        fieldInfoEl.innerHTML = '<i class="fa-solid fa-keyboard"></i> <span id="vk-field-name">Touch Keyboard</span>';

        previewEl = document.createElement('div');
        previewEl.className = 'vk-preview-text';
        previewEl.id = 'vk-preview';
        previewEl.textContent = '';

        const headerActions = document.createElement('div');
        headerActions.className = 'vk-header-actions';

        const prevBtn = document.createElement('button');
        prevBtn.type = 'button';
        prevBtn.className = 'vk-action-btn';
        prevBtn.innerHTML = '<i class="fa-solid fa-arrow-up"></i> Prev';
        prevBtn.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            focusSiblingInput(-1);
        });

        const nextBtn = document.createElement('button');
        nextBtn.type = 'button';
        nextBtn.className = 'vk-action-btn';
        nextBtn.innerHTML = '<i class="fa-solid fa-arrow-down"></i> Next';
        nextBtn.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            focusSiblingInput(1);
        });

        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'vk-action-btn vk-close-btn';
        closeBtn.innerHTML = '<i class="fa-solid fa-xmark"></i> Done';
        closeBtn.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            hideKeyboard();
        });

        headerActions.appendChild(prevBtn);
        headerActions.appendChild(nextBtn);
        headerActions.appendChild(closeBtn);

        header.appendChild(fieldInfoEl);
        header.appendChild(previewEl);
        header.appendChild(headerActions);

        // Body Grid
        bodyEl = document.createElement('div');
        bodyEl.className = 'vk-body';

        wrapper.appendChild(header);
        wrapper.appendChild(bodyEl);
        document.body.appendChild(wrapper);

        // Floating Toggle Button
        floatToggleBtn = document.createElement('button');
        floatToggleBtn.type = 'button';
        floatToggleBtn.id = 'vk-float-toggle';
        floatToggleBtn.className = 'vk-floating-toggle';
        floatToggleBtn.title = 'Open Virtual Keyboard';
        floatToggleBtn.innerHTML = '<i class="fa-solid fa-keyboard"></i>';
        floatToggleBtn.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (isVisible) {
                hideKeyboard();
            } else {
                if (activeInput && isInputElement(activeInput)) {
                    showKeyboard(activeInput);
                } else {
                    const firstInput = getEligibleInputs()[0];
                    if (firstInput) {
                        firstInput.focus();
                        showKeyboard(firstInput);
                    }
                }
            }
        });
        document.body.appendChild(floatToggleBtn);

        // Render Initial Layout
        renderKeys();

        // Attach Global Focus and Click Listeners
        attachInputListeners();
    }

    function isInputElement(el) {
        if (!el || !el.tagName) return false;
        const tag = el.tagName.toLowerCase();
        if (tag === 'textarea') return true;
        if (tag === 'input') {
            const type = (el.type || 'text').toLowerCase();
            const excluded = ['file', 'checkbox', 'radio', 'button', 'submit', 'reset', 'range', 'color', 'date', 'datetime-local', 'month', 'time', 'week'];
            return !excluded.includes(type);
        }
        return false;
    }

    function getEligibleInputs() {
        const elements = Array.from(document.querySelectorAll('input, textarea'));
        return elements.filter(el => {
            return isInputElement(el) && el.offsetParent !== null && !el.disabled && !el.readOnly;
        });
    }

    function attachInputListeners() {
        if (navigator.virtualKeyboard) {
            navigator.virtualKeyboard.overlaysContent = true;
        }

        function applyInputPolicy(el) {
            if (isInputElement(el)) {
                try {
                    el.setAttribute('virtualkeyboardpolicy', 'manual');
                } catch (e) {}
            }
        }

        document.querySelectorAll('input, textarea').forEach(applyInputPolicy);

        // Direct focus and pointer listeners
        document.addEventListener('focusin', function (e) {
            if (isInputElement(e.target)) {
                applyInputPolicy(e.target);
                showKeyboard(e.target);
            }
        }, true);

        document.addEventListener('pointerdown', function (e) {
            const path = e.composedPath ? e.composedPath() : [];
            const isInsideKeyboard = path.some(el => el === wrapper || el === floatToggleBtn);

            if (isInsideKeyboard) {
                return; // Let keyboard handlers process without interference
            }

            if (isInputElement(e.target)) {
                applyInputPolicy(e.target);
                showKeyboard(e.target);
                return;
            }

            // If user touched outside while keyboard is visible, hide it
            if (isVisible) {
                hideKeyboard();
            }
        }, true);
    }

    function showKeyboard(inputEl) {
        activeInput = inputEl;
        isVisible = true;
        wrapper.classList.add('vk-visible');
        floatToggleBtn.classList.add('vk-hidden-toggle');

        // Update Field Title
        const fieldNameEl = document.getElementById('vk-field-name');
        if (fieldNameEl) {
            let labelText = '';
            if (inputEl.id) {
                const label = document.querySelector(`label[for="${inputEl.id}"]`);
                if (label) labelText = label.textContent.trim().replace(/\*$/, '').trim();
            }
            if (!labelText && inputEl.placeholder) {
                labelText = inputEl.placeholder;
            }
            fieldNameEl.textContent = labelText || inputEl.name || 'Input Field';
        }

        // Auto-select numpad for numeric inputs
        if (inputEl.type === 'number' || (inputEl.inputMode && inputEl.inputMode === 'numeric')) {
            currentLayout = 'numpad';
        } else if (currentLayout === 'numpad') {
            currentLayout = 'qwerty';
        }

        updatePreview();
        renderKeys();

        // Scroll active input into view smoothly
        setTimeout(() => {
            if (activeInput) {
                activeInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }, 150);
    }

    function hideKeyboard() {
        isVisible = false;
        if (wrapper) wrapper.classList.remove('vk-visible');
        if (floatToggleBtn) floatToggleBtn.classList.remove('vk-hidden-toggle');
    }

    function updatePreview() {
        if (!previewEl) return;
        if (activeInput) {
            const isPassword = activeInput.type === 'password';
            const val = activeInput.value || '';
            previewEl.textContent = isPassword ? '•'.repeat(val.length) : val;
        } else {
            previewEl.textContent = '';
        }
    }

    function renderKeys() {
        if (!bodyEl) return;
        bodyEl.innerHTML = '';

        let layout = LAYOUT_QWERTY;
        if (currentLayout === 'symbols') layout = LAYOUT_SYMBOLS;
        else if (currentLayout === 'numpad') layout = LAYOUT_NUMPAD;

        if (currentLayout === 'numpad') {
            const numpadBox = document.createElement('div');
            numpadBox.className = 'vk-numpad-container';

            layout.forEach(rowKeys => {
                const rowEl = document.createElement('div');
                rowEl.className = 'vk-numpad-row';

                rowKeys.forEach(k => {
                    const keyBtn = createKeyButton(k, true);
                    rowEl.appendChild(keyBtn);
                });
                numpadBox.appendChild(rowEl);
            });

            bodyEl.appendChild(numpadBox);
            return;
        }

        layout.forEach(rowKeys => {
            const rowEl = document.createElement('div');
            rowEl.className = 'vk-row';

            rowKeys.forEach(k => {
                const keyBtn = createKeyButton(k, false);
                rowEl.appendChild(keyBtn);
            });
            bodyEl.appendChild(rowEl);
        });
    }

    function createKeyButton(key, isNumpad) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = isNumpad ? 'vk-key vk-numpad-key' : 'vk-key';

        let displayLabel = key;

        switch (key) {
            case 'shift':
                btn.classList.add('vk-key-special', 'vk-key-shift');
                if (isCapsLock) btn.classList.add('vk-caps-locked');
                else if (isShift) btn.classList.add('vk-shift-active');
                btn.innerHTML = '<i class="fa-solid fa-arrow-up"></i>';
                break;

            case 'backspace':
                btn.classList.add('vk-key-special', 'vk-key-backspace');
                btn.innerHTML = '<i class="fa-solid fa-delete-left"></i>';
                break;

            case 'space':
                btn.classList.add('vk-key-space');
                btn.innerHTML = '<span>␣ Space</span>';
                break;

            case 'enter':
                btn.classList.add('vk-key-enter');
                btn.innerHTML = '<i class="fa-solid fa-arrow-turn-down"></i> Enter';
                break;

            case 'tab':
                btn.classList.add('vk-key-special', 'vk-key-tab');
                btn.innerHTML = '<i class="fa-solid fa-right-to-bracket"></i> Tab';
                break;

            case 'clear':
                btn.classList.add('vk-key-special');
                btn.innerHTML = 'Clear';
                break;

            case '?123':
            case 'ABC':
                btn.classList.add('vk-key-special', 'vk-key-mode');
                btn.textContent = key;
                break;

            default:
                if (!isNumpad) {
                    displayLabel = (isShift || isCapsLock) ? key.toUpperCase() : key.toLowerCase();
                }
                btn.textContent = displayLabel;
                break;
        }

        function triggerPress(e) {
            e.preventDefault();
            e.stopPropagation();
            btn.classList.add('vk-active');
            handleKeyPress(key);
            setTimeout(() => btn.classList.remove('vk-active'), 120);
        }

        btn.addEventListener('pointerdown', triggerPress);

        return btn;
    }

    function handleKeyPress(key) {
        if (!activeInput) {
            const inputs = getEligibleInputs();
            if (inputs.length > 0) {
                activeInput = inputs[0];
                activeInput.focus();
            } else {
                return;
            }
        }

        switch (key) {
            case 'shift':
                if (isShift && !isCapsLock) {
                    isCapsLock = true;
                    isShift = false;
                } else if (isCapsLock) {
                    isCapsLock = false;
                    isShift = false;
                } else {
                    isShift = true;
                }
                renderKeys();
                break;

            case '?123':
                currentLayout = 'symbols';
                renderKeys();
                break;

            case 'ABC':
                currentLayout = 'qwerty';
                renderKeys();
                break;

            case 'backspace':
                deleteChar();
                break;

            case 'clear':
                clearInput();
                break;

            case 'space':
                insertChar(' ');
                break;

            case 'tab':
                focusSiblingInput(1);
                break;

            case 'enter':
                submitOrNext();
                break;

            default:
                let charToInsert = key;
                if (currentLayout === 'qwerty') {
                    charToInsert = (isShift || isCapsLock) ? key.toUpperCase() : key.toLowerCase();
                    if (isShift && !isCapsLock) {
                        isShift = false;
                        renderKeys();
                    }
                }
                insertChar(charToInsert);
                break;
        }

        updatePreview();
    }

    function insertChar(char) {
        if (!activeInput) return;

        const el = activeInput;
        const start = el.selectionStart ?? el.value.length;
        const end = el.selectionEnd ?? el.value.length;
        const val = el.value || '';

        el.value = val.substring(0, start) + char + val.substring(end);
        const newPos = start + char.length;

        try {
            el.setSelectionRange(newPos, newPos);
        } catch (e) {}

        // Dispatch input and change events for reactive form handlers
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function deleteChar() {
        if (!activeInput) return;

        const el = activeInput;
        const start = el.selectionStart ?? el.value.length;
        const end = el.selectionEnd ?? el.value.length;
        const val = el.value || '';

        if (start !== end) {
            el.value = val.substring(0, start) + val.substring(end);
            try {
                el.setSelectionRange(start, start);
            } catch (e) {}
        } else if (start > 0) {
            el.value = val.substring(0, start - 1) + val.substring(start);
            try {
                el.setSelectionRange(start - 1, start - 1);
            } catch (e) {}
        }

        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function clearInput() {
        if (!activeInput) return;
        activeInput.value = '';
        activeInput.dispatchEvent(new Event('input', { bubbles: true }));
        activeInput.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function focusSiblingInput(direction) {
        const inputs = getEligibleInputs();
        if (!inputs.length) return;

        let currentIndex = inputs.indexOf(activeInput);
        if (currentIndex === -1) currentIndex = 0;

        let targetIndex = currentIndex + direction;
        if (targetIndex >= inputs.length) targetIndex = 0;
        if (targetIndex < 0) targetIndex = inputs.length - 1;

        const target = inputs[targetIndex];
        if (target) {
            target.focus();
            showKeyboard(target);
        }
    }

    function submitOrNext() {
        if (!activeInput) {
            hideKeyboard();
            return;
        }

        const form = activeInput.closest('form');
        if (form) {
            const inputs = getEligibleInputs().filter(i => form.contains(i));
            const isLast = inputs.indexOf(activeInput) === inputs.length - 1;
            if (isLast) {
                hideKeyboard();
                form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
            } else {
                focusSiblingInput(1);
            }
        } else {
            hideKeyboard();
        }
    }

    // Expose API globally
    window.RashVirtualKeyboard = {
        init: initKeyboard,
        show: showKeyboard,
        hide: hideKeyboard
    };

    // Auto-init when DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initKeyboard);
    } else {
        initKeyboard();
    }
})();
