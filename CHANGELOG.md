# Changelog

## 3.7.2
- Added exact, tested top-level dependency pins in `requirements.lock`.
- Pinned Dask explicitly to stop pip dependency backtracking through many historical releases.
- Installation now occurs only on first launch or when the lock file/application version changes.
- Normal launches skip pip entirely after a quick import/hash check.
- Kept HDF4 in a separately pinned environment to avoid native HDF4/HDF5 library conflicts.
- Added persistent installation logging at `logs/install.log`.
- Launcher never updates Homebrew and never installs Python automatically.

## 3.7.1
- Uses an existing supported Python installation without invoking Homebrew.
