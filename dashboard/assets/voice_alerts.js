/**
 * Financial Contagion Monitor - Voice Alert System
 * Uses the Web Speech API to announce severe crises with spoken warnings.
 * The full-screen siren overlay is the single source of truth for spoken alerts.
 */

document.addEventListener("DOMContentLoaded", function() {
    console.log("[FCM] Voice Alert System initialized.");

    let lastSpokenSignature = "";
    let lastSpokenTime = 0;
    const COOLDOWN_MS = 15000;

    let speechUnlocked = false;
    let activeSirenSignature = "";
    let sirenVisible = false;

    document.addEventListener("click", function() {
        if (!speechUnlocked && "speechSynthesis" in window) {
            const unlockUtterance = new SpeechSynthesisUtterance("");
            unlockUtterance.volume = 0;
            window.speechSynthesis.speak(unlockUtterance);
            speechUnlocked = true;
            console.log("[FCM Voice] Speech unlocked via user interaction.");
        }
    }, { once: true });

    function stopSpeech(reason) {
        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
            console.log("[FCM Voice] Speech cancelled:", reason);
        }
    }

    function getVoices() {
        if (!("speechSynthesis" in window)) {
            return [];
        }
        return window.speechSynthesis.getVoices() || [];
    }

    function pickVoice(mode) {
        const voices = getVoices();
        if (!voices.length) {
            return null;
        }

        const englishVoices = voices.filter(function(voice) {
            return (voice.lang || "").toLowerCase().startsWith("en");
        });
        const pool = englishVoices.length ? englishVoices : voices;

        if (mode === "warning") {
            const warningVoice = pool.find(function(voice) {
                return /david|mark|george|guy|daniel|james/.test((voice.name || "").toLowerCase());
            });
            if (warningVoice) {
                return warningVoice;
            }
        }

        return pool.find(function(voice) {
            return (voice.lang || "").toLowerCase() === "en-in";
        }) || pool.find(function(voice) {
            return (voice.lang || "").toLowerCase() === "en-gb";
        }) || pool.find(function(voice) {
            return (voice.lang || "").toLowerCase() === "en-us";
        }) || pool[0];
    }

    function normalizeForSpeech(text) {
        if (!text) {
            return "";
        }

        return text
            .replace(/^CRISIS ORIGIN:\s*/i, "")
            .replace(/^AFFECTED CHAIN:\s*/i, "")
            .replace(/\bRBI\b/g, "R B I")
            .replace(/\bSRI\b/g, "systemic risk index")
            .replace(/&/g, "and")
            .replace(/[\u2013\u2014]/g, ". ")
            .replace(/\s+/g, " ")
            .trim();
    }

    function buildUtterance(text, options) {
        const utterance = new SpeechSynthesisUtterance();
        utterance.text = text;
        utterance.volume = 1.0;
        utterance.rate = options.rate;
        utterance.pitch = options.pitch;

        const voice = pickVoice(options.voiceMode || "default");
        if (voice) {
            utterance.voice = voice;
        }

        return utterance;
    }

    function queueSpeechPlan(plan, signature) {
        if (!("speechSynthesis" in window)) {
            console.warn("[FCM Voice] Speech synthesis not supported.");
            return;
        }

        const now = Date.now();
        if (signature === lastSpokenSignature && (now - lastSpokenTime) < COOLDOWN_MS) {
            return;
        }

        stopSpeech("queueing new alert");

        plan.forEach(function(part) {
            if (part && part.text) {
                window.speechSynthesis.speak(buildUtterance(part.text, part));
            }
        });

        lastSpokenSignature = signature;
        lastSpokenTime = now;
        console.log("[FCM Voice] Speaking alert:", signature);
    }

    function buildSirenSpeechPlan(crisisName, origin, affectedChain) {
        const cleanCrisis = normalizeForSpeech(crisisName);
        const cleanOrigin = normalizeForSpeech(origin);
        const cleanChain = normalizeForSpeech(affectedChain)
            .replace(/,\s*then\s+/gi, ". Then ")
            .replace(/,\s*and\s+/gi, ". And ");

        const plan = [
            { text: "Warning.", rate: 0.68, pitch: 0.45, voiceMode: "warning" },
            { text: "Financial Contagion Monitor has detected a critical systemic event.", rate: 0.82, pitch: 0.72, voiceMode: "default" },
        ];

        if (cleanCrisis) {
            plan.push({
                text: "Crisis type. " + cleanCrisis + ".",
                rate: 0.84,
                pitch: 0.72,
                voiceMode: "default",
            });
        }

        if (cleanOrigin && cleanOrigin.toLowerCase() !== "unknown") {
            plan.push({
                text: "Origin. " + cleanOrigin + ".",
                rate: 0.84,
                pitch: 0.72,
                voiceMode: "default",
            });
        }

        if (cleanChain) {
            plan.push({
                text: "Affected chain. " + cleanChain + ".",
                rate: 0.8,
                pitch: 0.72,
                voiceMode: "default",
            });
        }

        plan.push(
            { text: "Immediate supervisory attention is required.", rate: 0.8, pitch: 0.7, voiceMode: "default" },
            { text: "This is an automated alert from the R B I Financial Contagion Monitor.", rate: 0.82, pitch: 0.72, voiceMode: "default" }
        );

        return plan;
    }

    function bindDismissButton() {
        const dismissBtn = document.getElementById("siren-dismiss-btn");
        if (!dismissBtn || dismissBtn.dataset.voiceCancelBound === "true") {
            return;
        }

        dismissBtn.addEventListener("click", function() {
            stopSpeech("dismiss button clicked");
        });
        dismissBtn.dataset.voiceCancelBound = "true";
    }

    function watchSirenOverlay() {
        const checkSiren = setInterval(function() {
            const sirenOverlay = document.getElementById("siren-overlay");
            if (!sirenOverlay) {
                return;
            }

            clearInterval(checkSiren);
            bindDismissButton();

            const sirenObserver = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type !== "attributes" || mutation.attributeName !== "style") {
                        return;
                    }

                    const display = sirenOverlay.style.display;
                    const isVisible = Boolean(display && display !== "none");

                    if (!isVisible) {
                        sirenVisible = false;
                        activeSirenSignature = "";
                        stopSpeech("overlay hidden");
                        return;
                    }

                    const crisisEl = document.getElementById("siren-crisis-name");
                    const originEl = document.getElementById("siren-origin-bank");
                    const chainEl = document.getElementById("siren-affected-list");

                    const crisisName = crisisEl ? crisisEl.innerText.trim() : "";
                    const originName = originEl ? originEl.innerText.trim() : "";
                    const affectedChain = chainEl ? chainEl.innerText.trim() : "";
                    const signature = [crisisName, originName, affectedChain].join("|");

                    if (sirenVisible && signature === activeSirenSignature) {
                        return;
                    }

                    sirenVisible = true;
                    activeSirenSignature = signature;

                    if (!signature.replace(/\|/g, "").trim()) {
                        queueSpeechPlan([
                            { text: "Warning.", rate: 0.68, pitch: 0.45, voiceMode: "warning" },
                            { text: "The Financial Contagion Monitor has detected a critical systemic risk event. Immediate supervisory review is required.", rate: 0.82, pitch: 0.72, voiceMode: "default" },
                        ], "generic-warning");
                        return;
                    }

                    queueSpeechPlan(
                        buildSirenSpeechPlan(crisisName, originName, affectedChain),
                        signature
                    );
                });
            });

            sirenObserver.observe(sirenOverlay, { attributes: true, attributeFilter: ["style"] });
            console.log("[FCM Voice] Siren overlay observer attached.");
        }, 1000);
    }

    if (window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = function() {
            console.log("[FCM Voice] Voices loaded:", getVoices().length);
        };
    }

    watchSirenOverlay();
});
