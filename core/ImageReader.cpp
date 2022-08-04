// MIT License

// Copyright (c) 2022 jiwenchen(cjwbeyond@hotmail.com)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkNrrdImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"

#include "ImageReader.h"
#include "Logger.h"

using namespace MonkeyGL;

ImageReader::ImageReader()
{
}

ImageReader::~ImageReader()
{
}

bool ImageReader::Read(
    const char* szFile,
    std::shared_ptr<short>& pData,
    int dim[],
    double spacing[],
    Point3d& ptOrigin,
    Direction3d& dirX,
    Direction3d& dirY,
    Direction3d& dirZ
)
{
    itk::NrrdImageIOFactory::RegisterOneFactory();
    itk::NiftiImageIOFactory::RegisterOneFactory();
    itk::MetaImageIOFactory::RegisterOneFactory();

    using PixelType = short;
    constexpr unsigned int Dimension = 3;
    using ImageType = itk::Image<PixelType, Dimension>;

    using ReaderType = itk::ImageFileReader<ImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(szFile);

    try
    {
        reader->Update();
        ImageType::Pointer output = reader->GetOutput();

        const ImageType::SpacingType& spc = output->GetSpacing();
        const ImageType::PointType& origin = output->GetOrigin();
        const ImageType::SizeType size = output->GetLargestPossibleRegion().GetSize();
        const ImageType::DirectionType dir = output->GetDirection();

        dim[0] = size[0];
        dim[1] = size[1];
        dim[2] = size[2];
        spacing[0] = spc[0];
        spacing[1] = spc[1];
        spacing[2] = spc[2];
        ptOrigin = Point3d(origin[0], origin[1], origin[2]);
        dirX = Direction3d(dir[0][0], dir[0][1], dir[0][2]);
        dirY = Direction3d(dir[1][0], dir[1][1], dir[1][2]);
        dirZ = Direction3d(dir[2][0], dir[2][1], dir[2][2]);

        PixelType* pBuffer = output->GetBufferPointer();

        pData.reset(new short[ dim[0]*dim[1]*dim[2] ]);
        memcpy(pData.get(), pBuffer, dim[0]*dim[1]*dim[2]*sizeof(short));
    }
    catch (const itk::ExceptionObject & err)
    {
        Logger::Error("Failed to read file[%s], error[%s]", szFile, err.what());
        return false;
    }

    return true;
}


bool ImageReader::ReadMask(
    const char* szFile,
    std::shared_ptr<unsigned char>& pData,
    int& nWidth,
    int& nHeight,
    int& nDepth
)
{
    itk::NrrdImageIOFactory::RegisterOneFactory();
    itk::NiftiImageIOFactory::RegisterOneFactory();
    itk::MetaImageIOFactory::RegisterOneFactory();

    using PixelType = unsigned char;
    constexpr unsigned int Dimension = 3;
    using ImageType = itk::Image<PixelType, Dimension>;

    using ReaderType = itk::ImageFileReader<ImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(szFile);

    try
    {
        reader->Update();
        ImageType::Pointer output = reader->GetOutput();

        const ImageType::SpacingType& spc = output->GetSpacing();
        const ImageType::SizeType size = output->GetLargestPossibleRegion().GetSize();

        nWidth = size[0];
        nHeight = size[1];
        nDepth = size[2];

        PixelType* pBuffer = output->GetBufferPointer();

        pData.reset(new unsigned char[ nWidth*nHeight*nDepth ]);
        memcpy(pData.get(), pBuffer, nWidth*nHeight*nDepth*sizeof(unsigned char));
    }
    catch (const itk::ExceptionObject & err)
    {
        Logger::Error("Failed to read file[%s], error[%s]", szFile, err.what());
        return false;
    }
    
    return true;
    
}