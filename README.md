# Detecția colțurilor obiectelor din imagini (grayscale)

(1 student) Detecția colțurilor obiectelor din imagini (grayscale).
- se vor studia diverși algoritmi de detecție a colțurilor obiectelor;
- se va alege și implementa un algoritm de detecție a colțurilor obiectelor
(corner detection) din imagini (grayscale).

## Project Structure

```
ImageProcessingProject/
├── main.cpp              # Entry point, menu
├── common.h / common.cpp # Utility functions (file dialog, resize, etc.)
├── stdafx.h / stdafx.cpp # Precompiled header
├── targetver.h
├── CMakeLists.txt        # Build configuration
├── OpenCV/               # OpenCV 4.9.0 libraries
│   ├── include/
│   ├── lib/
│   └── dll/
└── Images/               # Test images
```

## Build (VS Code + CMake)

1. Install the following VS Code extensions:
   - **C/C++** (Microsoft)
   - **CMake Tools** (Microsoft)

2. Open the project folder in VS Code

3. `Ctrl+Shift+P` → **CMake: Configure** → select: `Visual Studio 2022 Release - amd64`

4. `Ctrl+Shift+P` → **CMake: Build** (or press F7)

5. Run: `Ctrl+F5`


