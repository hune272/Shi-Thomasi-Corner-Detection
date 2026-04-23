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
// Acces la un pixel dintr-un Mat cu strategia "clamp-to-edge":
// daca (y, x) este in afara imaginii, se intoarce valoarea celui mai apropiat
// pixel valid de pe margine (nu se adauga zero). Pastreaza intensitatea medie
// in apropierea marginilor si evita artefactele de tip "halo intunecat".
//
// Sablonizat dupa tipul de pixel:
//   - atClamp<float>(M, y, x)    pentru Mat-uri CV_32F (1 canal)
//   - atClamp<Point3f>(M, y, x)  pentru Mat-uri CV_32FC3 (3 canale, ex. Ixx/Ixy/Iyy)
// -----------------------------------------------------------------------------
template<typename T>
T atClamp(const Mat& img, int y, int x)
{
    if (y < 0) y = 0; else if (y >= img.rows) y = img.rows - 1;
    if (x < 0) x = 0; else if (x >= img.cols) x = img.cols - 1;
    return img.at<T>(y, x);
}

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
    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        // -----------------------------------------------------------------
        // Citirea imaginii sursa (color BGR, uchar).
        // -----------------------------------------------------------------
        Mat src = imread(fname);
        if (src.empty()) continue;

        const int H = src.rows;
        const int W = src.cols;

        // =================================================================
        // Pasul 1. Conversie manuala BGR -> grayscale.
        //
        //     Y(x, y) = 0.299 * R + 0.587 * G + 0.114 * B
        //
        // Stocam rezultatul intr-un Mat CV_32F pentru ca urmatorii pasi
        // (blur, gradient, structure matrix) lucreaza cu numere reale.
        // OpenCV stocheaza pixelii color in ordinea B, G, R.
        // =================================================================
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

        // =================================================================
        // Pasul 2. Gaussian blur 5x5, sigma = 1.0.
        //
        //   1) Construim nucleul (kernel) gaussian 5x5 dupa formula:
        //
        //        G(i, j) = (1 / (2*pi*sigma^2)) * exp(-(i^2 + j^2) / (2*sigma^2))
        //
        //      apoi normalizam astfel incat suma tuturor coeficientilor sa
        //      fie 1 (pastreaza intensitatea medie a imaginii).
        //
        //   2) Aplicam convolutie 2D pe imaginea grayscale. Pentru pixelii
        //      de pe margine, indicii care ar iesi din imagine sunt "prinsi"
        //      la cel mai apropiat pixel valid (clamp-to-edge) -- fara
        //      copyMakeBorder.
        // =================================================================
        const int   GK = 5;        // dimensiunea kernelului
        const int   GR = GK / 2;   // raza kernelului (= 2)
        const float sigma = 1.0f;

        float gKernel[GK][GK];
        {
            float sum = 0.0f;
            const float s2 = 2.0f * sigma * sigma;
            for (int i = -GR; i <= GR; ++i)
            {
                for (int j = -GR; j <= GR; ++j)
                {
                    const float v = std::exp(-(float)(i * i + j * j) / s2);
                    gKernel[i + GR][j + GR] = v;
                    sum += v;
                }
            }
            // Normalizare: suma coeficientilor = 1.
            for (int i = 0; i < GK; ++i)
                for (int j = 0; j < GK; ++j)
                    gKernel[i][j] /= sum;
        }

        // Convolutie 2D cu clamp-to-edge pe margini (atClamp).
        //   blurred(x, y) = sum_{i,j} gKernel[i,j] * gray[y+i, x+j]
        // Pentru indici din afara imaginii, atClamp intoarce valoarea celui
        // mai apropiat pixel valid, nu 0 -- astfel evitam intunecarea
        // marginilor.
        Mat blurred(H, W, CV_32F);
        for (int y = 0; y < H; ++y)
        {
            float* bRow = blurred.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                float acc = 0.0f;
                for (int i = -GR; i <= GR; ++i)
                    for (int j = -GR; j <= GR; ++j)
                        acc += gKernel[i + GR][j + GR] *
                               atClamp<float>(gray, y + i, x + j);
                bRow[x] = acc;
            }
        }
        //etapa 3
        //sobel kernels
        Mat sobelFiltered = Mat::zeros(H, W, CV_32FC2);
        float Gx[3][3] = { {-1, 0, 1},
                            {-2, 0, 2},
                            {-1, 0, 1} };
        float Gy[3][3] = { {1, 2, 1},
                            {0, 0, 0},
                            {-1, -2, -1} };   
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                float sumX = 0.0f;
                float sumY = 0.0f;
                for(int k = -1; k <= 1; k++){
                    for(int l = -1; l <= 1; l++){
                        sumX += Gx[k + 1][l + 1] * atClamp<float>(blurred, i + k, j + l);
                        sumY += Gy[k + 1][l + 1] * atClamp<float>(blurred, i + k, j + l);
                    }
                }
                //magnitudinea gradientului, stocare Ix si Iy in acelasi Mat pentru simplitate
                sobelFiltered.at<Point2f>(i, j) = Point2f(sumX, sumY);
            }
        }
        //etapa 4
        //construire matrice de structura
        Mat structureMatrix = Mat::zeros(H, W, CV_32FC3);
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                float Ix = sobelFiltered.at<Point2f>(i, j).x;
                float Iy = sobelFiltered.at<Point2f>(i, j).y;
                structureMatrix.at<Point3f>(i, j) = Point3f(Ix * Ix, Ix * Iy, Iy * Iy);
            }
        }
        //etapa 5
        //sumarea ponderata Gaussiana pe vecinatate pentru matricea de structura
        Mat weightedStructure = Mat::zeros(H, W, CV_32FC3);
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                Point3f acc = Point3f(0.0f, 0.0f, 0.0f);
                for(int k = -GR; k <= GR; k++){
                    for(int l = -GR; l <= GR; l++){
                        Point3f neighbor = atClamp<Point3f>(structureMatrix, i + k, j + l);
                        float weight = gKernel[k + GR][l + GR];
                        acc.x += weight * neighbor.x;
                        acc.y += weight * neighbor.y;
                        acc.z += weight * neighbor.z;
                    }
                }
                weightedStructure.at<Point3f>(i, j) = acc;
            }
        }
        
        //etapa 6
        //calcul valorii proprii prin formula inchisa
        //  trace = A + C
        //  det   = A*C - B*B
        //  lambda1,2 = trace/2 +/- sqrt(trace^2/4 - det)
        //  R(x, y)   = min(lambda1, lambda2) = trace/2 - sqrt(...)
        //
        // Protectie NaN: din cauza preciziei float, trace^2/4 - det poate
        // ajunge usor negativ (teoretic e intotdeauna >= 0 pentru matrice
        // simetrica). Aplicam max(0, ...) inainte de sqrt ca sa nu aparem
        // cu NaN in response, care strica maxResponse si threshold-ul.
        Mat response = Mat::zeros(H, W, CV_32F);
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                float A = weightedStructure.at<Point3f>(i, j).x;
                float B = weightedStructure.at<Point3f>(i, j).y;
                float C = weightedStructure.at<Point3f>(i, j).z;
                float trace = A + C;
                float det = A * C - B * B;
                float inside = trace * trace / 4 - det;
                if (inside < 0.0f) inside = 0.0f;
                float disc = std::sqrt(inside);
                float lambdaMin = trace / 2 - disc;
                response.at<float>(i, j) = lambdaMin;
            }
        }
        //etapa 7
        //thresholding: keep pixels with response > QUALITY_LEVEL * maxResponse
        const float QUALITY_LEVEL = 0.01f;
        float maxResponse = 0.0f;
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                const float v = response.at<float>(i, j);
                if(v > maxResponse) maxResponse = v;
            }
        }
        const float threshold = QUALITY_LEVEL * maxResponse;

        //etapa 8
        //non-maximum suppression pe fereastra (2*MIN_DISTANCE+1) x ...
        // Pentru fiecare pixel p cu response > threshold, verificam daca p
        // este maxim local in fereastra [y-D, y+D] x [x-D, x+D]. Asa evitam
        // comparatia O(n^2) intre toate perechile de candidati.
        const int MIN_DISTANCE = 10;
        std::vector<std::pair<Point2f, float>> nmsCorners;
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                const float v = response.at<float>(i, j);
                if(v <= threshold) continue;

                bool isMax = true;
                const int y0 = std::max(0, i - MIN_DISTANCE);
                const int y1 = std::min(H - 1, i + MIN_DISTANCE);
                const int x0 = std::max(0, j - MIN_DISTANCE);
                const int x1 = std::min(W - 1, j + MIN_DISTANCE);

                for(int yy = y0; yy <= y1 && isMax; yy++){
                    for(int xx = x0; xx <= x1; xx++){
                        if(yy == i && xx == j) continue;
                        const float nv = response.at<float>(yy, xx);
                        if(nv > v){ isMax = false; break; }
                        // tie-break: pentru valori egale, castiga primul in
                        // ordinea de scanare (evita duplicate pe platouri).
                        if(nv == v && (yy < i || (yy == i && xx < j))){
                            isMax = false; break;
                        }
                    }
                }

                if(isMax) nmsCorners.push_back(std::make_pair(Point2f((float)j, (float)i), v));
            }
        }
        //etapa 9
        //sortare dupa scor si pastrare top MAX_CORNERS
        const int MAX_CORNERS = 100;
        std::sort(nmsCorners.begin(), nmsCorners.end(), [](const std::pair<Point2f, float>& a, const std::pair<Point2f, float>& b){
            return a.second > b.second;
        });
        if(nmsCorners.size() > MAX_CORNERS){
            nmsCorners.resize(MAX_CORNERS);
        }
        //etapa 10
        //desenare cercuri rosii pe o copie a imaginii sursa si afisare
        Mat result = src.clone();
        for(size_t i = 0; i < nmsCorners.size(); i++){
            cv::circle(result, nmsCorners[i].first, 5, cv::Scalar(0, 0, 255), 2);
        }
        printf("Shi-Tomasi: maxR = %.4f, threshold = %.4f, colturi detectate = %d\n",
               maxResponse, threshold, (int)nmsCorners.size());

        imshow("Sursa", src);
        imshow("Colt detectat", result);
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
