// ShiTomasi.cpp
// -----------------------------------------------------------------------------
// Shi-Tomasi corner detector ("Good Features to Track", Shi & Tomasi, 1994)
// implemented from scratch, following the lab framework pattern.
//
// OpenCV is used ONLY for:
//   - image I/O and display (imread, imshow, imwrite, waitKey)
//   - drawing the detected corners (cv::circle)
//   - the Mat container as a float buffer (CV_32F)
//
// All image processing steps (grayscale conversion, Gaussian blur, Sobel
// gradients, structure matrix, eigenvalues, response, thresholding, non-maximum
// suppression) are hand-written.
// -----------------------------------------------------------------------------

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <direct.h>

// windows.h (pulled in via common.h) defines min/max as macros, which collide
// with std::min / std::max. Remove them so the STL versions are usable below.
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// -----------------------------------------------------------------------------
// testShiTomasi()
//
// Full Shi-Tomasi pipeline:
//   1. Grayscale conversion  (Y = 0.299 R + 0.587 G + 0.114 B)
//   2. Gaussian blur 5x5     (sigma = 1.0)
//   3. Sobel gradients       (Ix, Iy)
//   4. Structure matrix      (A = sum w*Ix^2,  B = sum w*Ix*Iy,  C = sum w*Iy^2)
//   5. Eigenvalues           (closed form for a 2x2 symmetric matrix)
//   6. Response              (R = min(lambda1, lambda2))
//   7. Thresholding          (R > QUALITY_LEVEL * max_R)
//   8. Non-maximum suppression in a MIN_DISTANCE x MIN_DISTANCE window
//   9. Sort by score, keep top MAX_CORNERS
//  10. Draw red circles on a copy of the source and display / save
// -----------------------------------------------------------------------------
void testShiTomasi()
{
    // --- Tunable parameters -------------------------------------------------
    const int   MAX_CORNERS   = 500;    // upper bound on reported corners
    const float QUALITY_LEVEL = 0.01f;  // fraction of max response used as threshold
    const int   MIN_DISTANCE  = 10;     // radius of the NMS window (pixels)

    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        Mat src = imread(fname);
        if (src.empty()) continue;

        const int H = src.rows;
        const int W = src.cols;

        // =====================================================================
        // Step 1. Manual grayscale conversion (BGR -> Y), stored as CV_32F.
        //         Y(x,y) = 0.299*R + 0.587*G + 0.114*B
        //         (OpenCV stores color images in BGR order.)
        // =====================================================================
        Mat gray(H, W, CV_32F);
        for (int y = 0; y < H; ++y)
        {
            const uchar* srcRow = src.ptr<uchar>(y);
            float*       gRow   = gray.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                const float B = (float)srcRow[3 * x + 0];
                const float G = (float)srcRow[3 * x + 1];
                const float R = (float)srcRow[3 * x + 2];
                gRow[x] = 0.299f * R + 0.587f * G + 0.114f * B;
            }
        }

        // Helper: clamp-to-edge index access for a CV_32F Mat.
        auto atClamp = [](const Mat& M, int y, int x) -> float {
            if (y < 0) y = 0; else if (y >= M.rows) y = M.rows - 1;
            if (x < 0) x = 0; else if (x >= M.cols) x = M.cols - 1;
            return M.at<float>(y, x);
        };

        // =====================================================================
        // Step 2. Gaussian blur with a 5x5 kernel, sigma = 1.0.
        //         G(i,j) = (1 / (2*pi*sigma^2)) * exp(-(i^2 + j^2) / (2*sigma^2))
        //         Kernel is normalised so that the sum of weights is 1.
        // =====================================================================
        const int   GK = 5;                  // kernel size
        const int   GR = GK / 2;             // kernel radius (= 2)
        const float sigma = 1.0f;
        float gKernel[GK][GK];
        {
            float sum = 0.0f;
            const float s2 = 2.0f * sigma * sigma;
            for (int i = -GR; i <= GR; ++i)
                for (int j = -GR; j <= GR; ++j)
                {
                    const float v = std::exp(-(float)(i*i + j*j) / s2);
                    gKernel[i + GR][j + GR] = v;
                    sum += v;
                }
            for (int i = 0; i < GK; ++i)
                for (int j = 0; j < GK; ++j)
                    gKernel[i][j] /= sum;
        }

        // 2D convolution with clamp-to-edge borders.
        Mat blurred(H, W, CV_32F);
        for (int y = 0; y < H; ++y)
        {
            float* bRow = blurred.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                float acc = 0.0f;
                for (int i = -GR; i <= GR; ++i)
                    for (int j = -GR; j <= GR; ++j)
                        acc += gKernel[i + GR][j + GR] * atClamp(gray, y + i, x + j);
                bRow[x] = acc;
            }
        }

        // =====================================================================
        // Step 3. Sobel gradients Ix, Iy via 3x3 kernels.
        //         Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        //         Sy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
        //         Ix = I * Sx,  Iy = I * Sy   (discrete convolution)
        // =====================================================================
        const int Sx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        const int Sy[3][3] = { {-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1} };

        Mat Ix(H, W, CV_32F);
        Mat Iy(H, W, CV_32F);
        for (int y = 0; y < H; ++y)
        {
            float* ixRow = Ix.ptr<float>(y);
            float* iyRow = Iy.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                float gx = 0.0f, gy = 0.0f;
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                    {
                        const float v = atClamp(blurred, y + i, x + j);
                        gx += (float)Sx[i + 1][j + 1] * v;
                        gy += (float)Sy[i + 1][j + 1] * v;
                    }
                ixRow[x] = gx;
                iyRow[x] = gy;
            }
        }

        // =====================================================================
        // Step 4. Structure matrix M = [[A, B], [B, C]] per pixel, with a 3x3
        //         Gaussian-weighted window w:
        //             A = sum  w * Ix^2
        //             B = sum  w * Ix * Iy
        //             C = sum  w * Iy^2
        //         The small 3x3 Gaussian below is separable but we build it
        //         directly; weights sum to 1.
        // =====================================================================
        const float w3[3][3] = {
            { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f },
            { 2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f },
            { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f }
        };

        Mat A(H, W, CV_32F);
        Mat B(H, W, CV_32F);
        Mat C(H, W, CV_32F);
        for (int y = 0; y < H; ++y)
        {
            float* aRow = A.ptr<float>(y);
            float* bRow = B.ptr<float>(y);
            float* cRow = C.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                float a = 0.0f, b = 0.0f, c = 0.0f;
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                    {
                        const float ix = atClamp(Ix, y + i, x + j);
                        const float iy = atClamp(Iy, y + i, x + j);
                        const float wt = w3[i + 1][j + 1];
                        a += wt * ix * ix;
                        b += wt * ix * iy;
                        c += wt * iy * iy;
                    }
                aRow[x] = a;
                bRow[x] = b;
                cRow[x] = c;
            }
        }

        // =====================================================================
        // Step 5 & 6. Eigenvalues of the symmetric 2x2 matrix [[A, B], [B, C]]
        //         via the closed-form formula:
        //             trace = A + C
        //             det   = A*C - B*B
        //             disc  = sqrt(max(0, (trace/2)^2 - det))
        //             lambda1 = trace/2 + disc
        //             lambda2 = trace/2 - disc
        //         Shi-Tomasi response: R = min(lambda1, lambda2) = trace/2 - disc.
        // =====================================================================
        Mat R(H, W, CV_32F);
        float maxR = 0.0f;
        for (int y = 0; y < H; ++y)
        {
            const float* aRow = A.ptr<float>(y);
            const float* bRow = B.ptr<float>(y);
            const float* cRow = C.ptr<float>(y);
            float*       rRow = R.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                const float a = aRow[x], b = bRow[x], c = cRow[x];
                const float halfTr = 0.5f * (a + c);
                const float det    = a * c - b * b;
                float       inside = halfTr * halfTr - det;
                if (inside < 0.0f) inside = 0.0f;     // guard against FP noise
                const float disc   = std::sqrt(inside);
                const float lambdaMin = halfTr - disc;
                rRow[x] = lambdaMin;
                if (lambdaMin > maxR) maxR = lambdaMin;
            }
        }

        // =====================================================================
        // Step 7. Thresholding: keep only pixels with R(x,y) > Q * maxR.
        // =====================================================================
        const float threshold = QUALITY_LEVEL * maxR;

        // =====================================================================
        // Step 8. Non-maximum suppression in a (2*MIN_DISTANCE+1) x ... window.
        //         A pixel survives only if it is strictly greater than every
        //         other pixel inside this window. Ties on the boundary are
        //         broken by (y, x) lexicographic order to avoid duplicates.
        // =====================================================================
        std::vector<std::pair<float, Point>> corners;
        corners.reserve(4096);
        const int RAD = MIN_DISTANCE;

        for (int y = 0; y < H; ++y)
        {
            const float* rRow = R.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                const float v = rRow[x];
                if (v <= threshold) continue;

                bool isMax = true;
                const int y0 = std::max(0, y - RAD);
                const int y1 = std::min(H - 1, y + RAD);
                const int x0 = std::max(0, x - RAD);
                const int x1 = std::min(W - 1, x + RAD);

                for (int yy = y0; yy <= y1 && isMax; ++yy)
                {
                    const float* nRow = R.ptr<float>(yy);
                    for (int xx = x0; xx <= x1; ++xx)
                    {
                        if (yy == y && xx == x) continue;
                        const float nv = nRow[xx];
                        if (nv > v) { isMax = false; break; }
                        // tie-break: equal neighbour earlier in scan order wins
                        if (nv == v && (yy < y || (yy == y && xx < x)))
                        {
                            isMax = false;
                            break;
                        }
                    }
                }

                if (isMax) corners.emplace_back(v, Point(x, y));
            }
        }

        // =====================================================================
        // Step 9. Sort by score descending, keep top MAX_CORNERS.
        // =====================================================================
        std::sort(corners.begin(), corners.end(),
                  [](const std::pair<float, Point>& a,
                     const std::pair<float, Point>& b) {
                      return a.first > b.first;
                  });
        if ((int)corners.size() > MAX_CORNERS)
            corners.resize(MAX_CORNERS);

        printf("Shi-Tomasi: maxR = %.4f, threshold = %.4f, corners kept = %d\n",
               maxR, threshold, (int)corners.size());

        // =====================================================================
        // Step 10. Draw red circles on a copy of the original image.
        //          cv::circle is the only drawing primitive used.
        // =====================================================================
        Mat result = src.clone();
        for (const auto& kp : corners)
            circle(result, kp.second, 5, Scalar(0, 0, 255), 1, LINE_AA);

        // =====================================================================
        // Display and save as "<original>_corners.<ext>".
        // =====================================================================
        imshow("Shi-Tomasi Corners", result);

        {
            char outName[MAX_PATH];
            const char* dot = strrchr(fname, '.');
            if (dot != nullptr)
            {
                const int stemLen = (int)(dot - fname);
                snprintf(outName, MAX_PATH, "%.*s_corners%s",
                         stemLen, fname, dot);
            }
            else
            {
                snprintf(outName, MAX_PATH, "%s_corners.bmp", fname);
            }
            imwrite(outName, result);
            printf("Saved: %s\n", outName);
        }

        waitKey();
    }
}

// -----------------------------------------------------------------------------
// Standalone menu for exercising the Shi-Tomasi detector. This file is
// self-contained: to use it, compile ShiTomasi.cpp together with common.cpp and
// stdafx.cpp (replace OpenCVApplication.cpp in the CMake target, or add a
// second executable target). The symbol `projectPath` is defined here so no
// external file is required.
// -----------------------------------------------------------------------------
wchar_t* projectPath;

// Try to locate the project's "Images" folder and make it the current working
// directory, so the Win32 file-open dialog starts there. Search candidates:
//   1. the directory of the running executable, walking upwards
//   2. the initial working directory, walking upwards
// The first "Images" subdirectory found wins.
static bool setImagesFolderAsCwd()
{
    auto tryChdirImagesFrom = [](const char* start) -> bool {
        char path[MAX_PATH];
        strncpy(path, start, MAX_PATH - 1);
        path[MAX_PATH - 1] = '\0';

        for (int depth = 0; depth < 6; ++depth)
        {
            char candidate[MAX_PATH];
            snprintf(candidate, MAX_PATH, "%s\\Images", path);
            DWORD attr = GetFileAttributesA(candidate);
            if (attr != INVALID_FILE_ATTRIBUTES &&
                (attr & FILE_ATTRIBUTE_DIRECTORY))
            {
                return SetCurrentDirectoryA(candidate) != 0;
            }
            // go up one level
            char* slash = strrchr(path, '\\');
            if (slash == nullptr) break;
            *slash = '\0';
        }
        return false;
    };

    // 1) try starting from the directory containing this .exe
    char exePath[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, exePath, MAX_PATH);
    if (n > 0 && n < MAX_PATH)
    {
        char* slash = strrchr(exePath, '\\');
        if (slash != nullptr)
        {
            *slash = '\0';
            if (tryChdirImagesFrom(exePath)) return true;
        }
    }

    // 2) fall back to the current working directory
    char cwd[MAX_PATH];
    if (_getcwd(cwd, MAX_PATH) != nullptr)
    {
        if (tryChdirImagesFrom(cwd)) return true;
    }

    return false;
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

    if (setImagesFolderAsCwd())
    {
        char cwd[MAX_PATH];
        if (_getcwd(cwd, MAX_PATH) != nullptr)
            printf("[info] working directory set to: %s\n", cwd);
    }
    else
    {
        printf("[info] 'Images' folder not found nearby; "
               "the file dialog will open in the default location.\n");
    }

    projectPath = _wgetcwd(0, 0);

    int op;
    do
    {
        system("cls");
        destroyAllWindows();
        printf("========================================\n");
        printf("  Shi-Tomasi Corner Detector (from scratch)\n");
        printf("========================================\n");
        printf(" 1 - Run Shi-Tomasi on a selected image\n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        if (scanf("%d", &op) != 1) { op = 0; }

        switch (op)
        {
        case 1:
            testShiTomasi();
            break;
        case 0:
            break;
        default:
            printf("Unknown option.\n");
            break;
        }
    } while (op != 0);

    return 0;
}
