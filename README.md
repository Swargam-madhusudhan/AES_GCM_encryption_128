AES-GCM 128 Encryption Implementation

Sequential Code — Course Project

Build & Execution Instructions
# Clone the repository
git clone https://github.com/Swargam-madhusudhan/AES_GCM_encryption_128.git
cd AES_GCM_encryption_128

# Create a build directory
mkdir build
cd build

# Configure and build the project
cmake ../
make

# Run the executable with sample input/output files
./AES_Encryption_GCM_Mode -i ../Dataset/1/PT.dat -e ../Dataset/1/CT.dat -t vector
