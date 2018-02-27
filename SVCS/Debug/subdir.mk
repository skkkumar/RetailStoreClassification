################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../TEST2.cpp \
../Test.cpp \
../Training.cpp \
../camera_calibration.cpp \
../collect_CheckerImg_for_calib.cpp 

OBJS += \
./TEST2.o \
./Test.o \
./Training.o \
./camera_calibration.o \
./collect_CheckerImg_for_calib.o 

CPP_DEPS += \
./TEST2.d \
./Test.d \
./Training.d \
./camera_calibration.d \
./collect_CheckerImg_for_calib.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -I/usr/local/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


