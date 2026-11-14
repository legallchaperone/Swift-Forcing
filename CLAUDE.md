# Project Overview
Apply Swift (https://www.alphaxiv.org/pdf/2610.25844) on the ODE Initialization step of Self Forcing.

Original ODE init: 
$$
\mathcal{L}_{\text{ODE}}^{\text{Swift}}
= \mathbb{E}\,\Big\| G_\phi\big(x_t^{i},\, x_0^{<i},\, t\big)
-  y|^2
$$
Swift(Ours): 
$$
\mathcal{L}_{\text{ODE}}^{\text{Swift}}
= \mathbb{E}\,\Big\| G_\phi\big(x_t^{i},\, x_0^{<i},\, t\big)
- \Big[\, y + {\alpha\,\big(y - G_{\text{weak}}(x_t^{i},\, x_0^{<i},\, t)\big)} \Big] \Big\|^2
$$

where the weak model will be an early check point of the few step student model here.

# General Guidelines:
- This codebase is yours: what it's for and how it's built are your call, and you'll keep building on it.

# Communicating clearly
Write for an intelligent but low-context reader. The person reading has not seen your tool calls, your reasoning or the files you opened, and may be switching in from other work. Messages should generally be self-contained: the reader should understand what you mean without having to dig for other context. Put the key points in the first few sentences. Practice progressive disclosure in your sentence order, and do not use undefined references and ideas in line 1 that won't make sense until the reader reaches line 12.

Use plain words, and write in clear, understandable full sentences. Do not use a term or abbreviation that you made up during the work. Say what the thing is in a way any reader would understand. Write mostly in the style of ASD-STE100 Simplified Technical English: short sentences, one idea per sentence, active voice, and consistently using the same word for the same thing. Use a paragraph of flowing prose where ideas connect, and keep bullets for lists of separate items. Leave out unnecessary details unless the reader asks and practice concision, but do ensure that your messages have the key context. Aim to make your user-facing text as human-readable and clear as possible.