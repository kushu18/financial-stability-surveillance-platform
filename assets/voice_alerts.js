/**
 * Financial Contagion Monitor - Voice Alert System v3
 * Uses the Web Speech API to announce severe crises with spoken warnings.
 *
 * Voice Design:
 *   - "WARNING" is spoken with a deep, commanding tone (low pitch, slow rate)
 *   - Contagion chain is narrated naturally:
 *       "Axis Bank is affected. SBI will be affected next.
 *        After that, HDFC will be affected..."
 *   - No robotic arrow symbols — pure human narration.
 *
 * DISMISSAL CONTRACT (works for ALL logins — RBI, affected banks, everyone):
 *   - Clicking any dismiss/acknowledge/reset button kills ALL speech instantly
 *     and sets a global "dismissed" flag.
 *   - Once dismissed, NO new speech fires until the siren overlay fully resets
 *     (returns to display:none), which re-arms the system for the next crisis.
 */

document.addEventListener("DOMContentLoaded", function() {
    console.log("[FCM] Voice Alert System v3 initialized.");

    let lastSpokenMessage = "";
    let lastSpokenTime = 0;
    const COOLDOWN_MS = 20000; // Don't repeat same alert within 20 seconds

    let speechUnlocked = false;

    // ── GLOBAL DISMISS FLAG ──────────────────────────────────────────────
    let dismissed = false;

    // Unlock speech synthesis on first user interaction
    document.addEventListener('click', function(e) {
        if (!speechUnlocked && 'speechSynthesis' in window) {
            const u = new SpeechSynthesisUtterance('');
            u.volume = 0;
            window.speechSynthesis.speak(u);
            speechUnlocked = true;
            console.log("[FCM Voice] Speech unlocked via user interaction.");
        }

        // Handle voice dismissal — check if any dismiss/acknowledge/reset button was clicked
        let target = e.target;
        let isDismiss = false;

        while (target && target !== document.body) {
            if (target.tagName === 'BUTTON') {
                const text = (target.innerText || "").toUpperCase();
                if (target.id === 'siren-dismiss-btn' || target.id === 'ack-btn' || target.id === 'reset-shock-btn' ||
                    text.includes('ACKNOWLEDGE') || text.includes('DISMISS') || text.includes('RESET')) {
                    isDismiss = true;
                }
                break;
            }
            target = target.parentElement;
        }

        if (isDismiss) {
            killAllSpeech();
            dismissed = true;
            console.log("[FCM Voice] Alert dismissed — all speech killed, blocked until crisis resets.");
        }
    });

    // ── Kill ALL Speech ───────────────────────────────────────────────────
    function killAllSpeech() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            // Chrome queues utterances; triple-cancel to be absolutely sure
            setTimeout(function() { window.speechSynthesis.cancel(); }, 50);
            setTimeout(function() { window.speechSynthesis.cancel(); }, 200);
            setTimeout(function() { window.speechSynthesis.cancel(); }, 500);
        }
    }

    // ── Get Best Voice ────────────────────────────────────────────────────
    function getBestVoice() {
        const voices = window.speechSynthesis.getVoices();
        // Prefer deep, authoritative voices
        return voices.find(v => v.name.includes('Google UK English Male'))
            || voices.find(v => v.name.includes('Microsoft David'))
            || voices.find(v => v.name.includes('Microsoft Mark'))
            || voices.find(v => v.name.includes('Daniel'))
            || voices.find(v => v.lang === 'en-GB' && v.name.toLowerCase().includes('male'))
            || voices.find(v => v.lang === 'en-GB')
            || voices.find(v => v.lang === 'en-US')
            || voices.find(v => v.lang.startsWith('en'))
            || null;
    }

    // ── Speak WARNING — Deep, Commanding (Kratos-style) ──────────────────
    function speakWarning(onEndCallback) {
        if (dismissed || !('speechSynthesis' in window)) return;

        const warningMsg = new SpeechSynthesisUtterance();
        warningMsg.text = "WARNING! ... WARNING! ... WARNING!";
        warningMsg.volume = 1.0;
        warningMsg.rate = 0.55;    // Very slow — heavy and commanding
        warningMsg.pitch = 0.3;    // Very deep — Kratos-like gravitas
        
        const voice = getBestVoice();
        if (voice) warningMsg.voice = voice;

        warningMsg.onend = function() {
            if (!dismissed && onEndCallback) {
                onEndCallback();
            }
        };

        window.speechSynthesis.speak(warningMsg);
        console.log("[FCM Voice] Speaking WARNING (deep tone).");
    }

    // ── Speak Informational Message — Normal, Clear Human Voice ──────────
    function speakInfo(text) {
        if (dismissed || !('speechSynthesis' in window)) return;

        const msg = new SpeechSynthesisUtterance();
        msg.text = text;
        msg.volume = 1.0;
        msg.rate = 0.90;    // Slightly slow for clarity
        msg.pitch = 0.85;   // Natural human pitch
        
        const voice = getBestVoice();
        if (voice) msg.voice = voice;

        window.speechSynthesis.speak(msg);
        console.log("[FCM Voice] Speaking info:", text.substring(0, 80) + "...");
    }

    // ── Full Alert: WARNING + Natural Chain Narration ─────────────────────
    function speakFullAlert(text) {
        if (dismissed) {
            console.log("[FCM Voice] Speech blocked — dismissed flag is active.");
            return;
        }
        if (!('speechSynthesis' in window)) return;

        // Prevent spamming the same message
        const now = Date.now();
        if (text === lastSpokenMessage && (now - lastSpokenTime) < COOLDOWN_MS) {
            return;
        }

        // Cancel any ongoing speech before starting new
        window.speechSynthesis.cancel();

        lastSpokenMessage = text;
        lastSpokenTime = now;

        // Speak deep WARNING first, then the informational message
        speakWarning(function() {
            speakInfo(text);
        });
    }

    // ── Build Natural Contagion Chain Message ─────────────────────────────
    function buildSirenMessage(affectedBanks, origin, crisisName) {
        let message = "The Financial Contagion Monitor has detected a critical systemic event. ";

        if (crisisName) {
            // Clean up crisis name — remove technical suffixes
            let cleanCrisis = crisisName.replace(/\s*—\s*/g, ", caused by ");
            message += "Crisis identified: " + cleanCrisis + ". ";
        }

        if (origin && origin !== "Unknown" && origin !== "") {
            // Clean origin — remove "CRISIS ORIGIN: " prefix if present
            let cleanOrigin = origin.replace("CRISIS ORIGIN: ", "");
            message += "The crisis originated at " + cleanOrigin + ". ";
        }

        // Build natural contagion chain narration
        if (affectedBanks && affectedBanks.length > 0) {
            // Clean bank names — remove arrow symbols and trim
            let cleanBanks = affectedBanks.map(function(b) {
                return b.replace(/→/g, "").replace(/>/g, "").trim();
            }).filter(function(b) { return b.length > 0; });

            if (cleanBanks.length === 1) {
                message += cleanBanks[0] + " is critically affected and requires immediate attention. ";
            } else if (cleanBanks.length === 2) {
                message += cleanBanks[0] + " is affected. " + cleanBanks[1] + " will be affected next. ";
            } else {
                // Natural chain narration
                message += cleanBanks[0] + " is affected first. ";
                for (let i = 1; i < cleanBanks.length; i++) {
                    if (i === 1) {
                        message += cleanBanks[i] + " will be affected next. ";
                    } else if (i === cleanBanks.length - 1) {
                        message += "And finally, " + cleanBanks[i] + " will also be affected. ";
                    } else if (i <= 5) {
                        message += "After that, " + cleanBanks[i] + " will be affected. ";
                    } else if (i === 6) {
                        let remaining = cleanBanks.length - i;
                        message += "And " + remaining + " more banks are at risk. ";
                        break;
                    }
                }
            }
        }

        message += "Immediate supervisory attention is required. ";
        message += "This is an automated alert from the Reserve Bank of India, Financial Contagion Monitor.";

        return message;
    }

    // ── Monitor Siren Overlay ─────────────────────────────────────────────
    function watchSirenOverlay() {
        const checkSiren = setInterval(function() {
            const sirenOverlay = document.getElementById("siren-overlay");
            if (sirenOverlay) {
                clearInterval(checkSiren);

                const sirenObserver = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                            const display = sirenOverlay.style.display;
                            if (display && display !== "none") {
                                // Siren is active — only speak if NOT dismissed
                                if (!dismissed) {
                                    const bankListEl = document.getElementById("siren-affected-list");
                                    const crisisEl = document.getElementById("siren-crisis-name");
                                    const originEl = document.getElementById("siren-origin-bank");

                                    let banks = [];
                                    let crisisName = "";
                                    let originName = "";

                                    if (crisisEl) {
                                        crisisName = crisisEl.innerText.trim();
                                    }
                                    if (originEl) {
                                        originName = originEl.innerText.trim();
                                    }
                                    // Read the affected chain text — now already in natural language
                                    let chainText = "";
                                    if (bankListEl) {
                                        chainText = bankListEl.innerText.trim();
                                        chainText = chainText.replace("AFFECTED CHAIN:", "").trim();
                                    }

                                    // Build the full spoken message
                                    let fullMessage = "The Financial Contagion Monitor has detected a critical systemic event. ";
                                    if (crisisName) {
                                        fullMessage += "Crisis identified: " + crisisName.replace(/\s*—\s*/g, ", caused by ") + ". ";
                                    }
                                    if (originName && originName !== "Unknown") {
                                        let cleanOrigin = originName.replace("CRISIS ORIGIN: ", "");
                                        fullMessage += "The crisis originated at " + cleanOrigin + ". ";
                                    }
                                    if (chainText.length > 0) {
                                        fullMessage += chainText + ". ";
                                    } else {
                                        fullMessage += "Multiple banks are under severe stress. ";
                                    }
                                    fullMessage += "Immediate supervisory attention is required. ";
                                    fullMessage += "This is an automated alert from the Reserve Bank of India, Financial Contagion Monitor.";

                                    speakFullAlert(fullMessage);
                                }
                            } else if (display === "none") {
                                // Siren hidden — kill lingering speech and RE-ARM for next crisis
                                killAllSpeech();
                                dismissed = false;
                                lastSpokenMessage = "";
                                console.log("[FCM Voice] Siren overlay hidden — system re-armed for next crisis.");
                            }
                        }
                    });
                });

                sirenObserver.observe(sirenOverlay, { attributes: true, attributeFilter: ['style'] });
                console.log("[FCM Voice] Siren overlay observer attached.");
            }
        }, 1000);
    }

    // ── Monitor Crisis Banner ─────────────────────────────────────────────
    function watchCrisisBanner() {
        const checkBanner = setInterval(function() {
            const banner = document.getElementById("crisis-banner");
            const bannerText = document.getElementById("crisis-banner-text");
            if (banner && bannerText) {
                clearInterval(checkBanner);

                let lastBannerText = "";

                const bannerObserver = new MutationObserver(function() {
                    if (dismissed) return;

                    if (banner.style.display !== "none" && bannerText.innerText) {
                        const text = bannerText.innerText.trim();
                        if (text.length > 5 && text !== lastBannerText) {
                            lastBannerText = text;
                            if (text.includes("CRISIS") || text.includes("PANDEMIC") || text.includes("MORATORIUM")) {
                                speakFullAlert("Crisis alert. " + text);
                            }
                        }
                    } else if (banner.style.display === "none") {
                        lastBannerText = "";
                        killAllSpeech();
                    }
                });

                bannerObserver.observe(bannerText, { childList: true, characterData: true, subtree: true });
                bannerObserver.observe(banner, { attributes: true, attributeFilter: ['style'] });
                console.log("[FCM Voice] Crisis banner observer attached.");
            }
        }, 1000);
    }

    // ── Monitor RBI Flag Banner ───────────────────────────────────────────
    function watchRBIFlag() {
        const checkRBI = setInterval(function() {
            const rbiFlag = document.getElementById("rbi-flag-banner");
            if (rbiFlag) {
                clearInterval(checkRBI);

                let lastRBIText = "";

                const rbiObserver = new MutationObserver(function() {
                    if (dismissed) return;

                    if (rbiFlag.style.display !== "none") {
                        const text = rbiFlag.innerText.trim();
                        if (text && text.includes("RBI INTERVENTION") && text !== lastRBIText) {
                            lastRBIText = text;
                            speakFullAlert(
                                "Critical alert. The Reserve Bank of India intervention is required. " +
                                "Domestically systemically important banks have been affected. " +
                                "Immediate action is recommended."
                            );
                        }
                    } else {
                        lastRBIText = "";
                    }
                });

                rbiObserver.observe(rbiFlag, { attributes: true, attributeFilter: ['style'], childList: true, subtree: true });
                console.log("[FCM Voice] RBI flag observer attached.");
            }
        }, 1000);
    }

    // ── Initialize ────────────────────────────────────────────────────────
    if (window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = function() {
            console.log("[FCM Voice] Voices loaded:", window.speechSynthesis.getVoices().length);
        };
    }

    watchSirenOverlay();
    watchCrisisBanner();
    watchRBIFlag();
});
