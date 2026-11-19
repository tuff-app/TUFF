# TUFF 11.2.3

## Settings fits at the smallest window size

- With the window at its smallest width and the sidebar open, the Settings tab bar was wider than the pane beside the sidebar. The whole window's content shifted sideways and both edges were clipped, most visibly on the Generation and Benchmarks tabs. The tab bar now tightens the space around its icons when there is no room, and shows icons alone as a last resort.
- The tab bar picks its layout for its longest title, so the spacing around the icons does not change when you switch tabs.
- At the same width the Zoom row in Settings > General had no room for its title beside the five choices, and the title wrapped one letter per line. The title now sits above the choices there.
- Nothing changes at wider sizes: Settings at 1,080 points wide renders pixel for pixel as before.
- Two labels now follow the capitalization used everywhere else: Load Model at Launch in the Settings menu, and "No Apple models found on this Mac" in Models.

## Compatibility

This patch uses the existing inference engines. Models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
