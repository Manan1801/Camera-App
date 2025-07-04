import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  StyleSheet,
  TouchableOpacity,
  Text,
  Image,
  Alert,
} from 'react-native';
import { Camera, useCameraDevices } from 'react-native-vision-camera';
import { RNFFmpeg } from 'react-native-ffmpeg';
import RNFS from 'react-native-fs';

function App() {
  const camera = useRef<Camera>(null);
  const devices = useCameraDevices();
  const device = devices.back;

  const [showCamera, setShowCamera] = useState(false);
  const [imageSource, setImageSource] = useState('');
  const [videoPath, setVideoPath] = useState('');

  useEffect(() => {
    async function getPermission() {
      const cameraPermission = await Camera.requestCameraPermission();
      const microphonePermission = await Camera.requestMicrophonePermission();

      if (cameraPermission !== 'authorized' || microphonePermission !== 'authorized') {
        Alert.alert('Permissions Denied', 'Camera and microphone permissions are required.');
      }
    }
    getPermission();
  }, []);

  const capturePhoto = async () => {
    if (camera.current !== null) {
      const photo = await camera.current.takePhoto({});
      setImageSource(photo.path);
      setShowCamera(false);
      console.log('Photo saved to:', photo.path);
    }
  };

  const captureVideo = async () => {
    if (camera.current !== null) {
      await camera.current.startRecording({
        onRecordingFinished: (video) => {
          console.log('Video recording finished:', video);

          const formData = new FormData();
          formData.append('video', {
            uri: video.path,
            type: 'video/mp4',
            name: 'video.mp4',
          });
          fetch('http://your-flask-server/upload', {
            method: 'POST',
            body: formData,
          })
            .then(response => response.json())
            .then(data => console.log('Upload success:', data))
            .catch(error => console.error('Upload error:', error));

          console.log('Video saved to:', video.path);
          setVideoPath(video.path);
          setShowCamera(false);
        },
        onRecordingError: (error) => {
          console.error('Recording error:', error);
        },
      });
      console.log('Recording started');
    }
  };

  const stopRecording = async () => {
    if (camera.current !== null) {
      await camera.current.stopRecording();
      console.log('Recording stopped');
    }
  };

  const extractFrames = async () => {
    if (!videoPath) {
      Alert.alert('No video found', 'Please record a video first.');
      return;
    }

    const outputDir = `${RNFS.ExternalStorageDirectoryPath}/frames`;
    if (!(await RNFS.exists(outputDir))) {
      await RNFS.mkdir(outputDir);
    }

    const command = `-i ${videoPath} -vf fps=60 ${outputDir}/frame%d.png`;

    try {
      await RNFFmpeg.execute(command);
      Alert.alert('Frames Extracted', `Frames saved to: ${outputDir}`);
      console.log('Frames extracted successfully');
    } catch (error) {
      console.error('Error extracting frames:', error);
      Alert.alert('Error', 'Failed to extract frames.');
    }
  };

  if (device == null) {
    return <Text>Camera not available</Text>;
  }

  return (
    <View style={styles.container}>
      {showCamera ? (
        <>
          <Camera
            ref={camera}
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={showCamera}
            photo={true}
            video={true}
            fps={60}
          />

          <View style={styles.buttonContainer}>
            <TouchableOpacity style={styles.camButton} onPress={capturePhoto}>
              <Text style={{ color: 'white' }}>Capture Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.camButton} onPress={captureVideo}>
              <Text style={{ color: 'white' }}>Start Video</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.camButton} onPress={stopRecording}>
              <Text style={{ color: 'white' }}>Stop Video</Text>
            </TouchableOpacity>
          </View>
        </>
      ) : (
        <>
          {imageSource !== '' ? (
            <Image
              style={styles.image}
              source={{
                uri: `file://${imageSource}`,
              }}
            />
          ) : null}

          <View style={styles.backButton}>
            <TouchableOpacity
              style={styles.backButtonStyle}
              onPress={() => setShowCamera(true)}
            >
              <Text style={{ color: 'white', fontWeight: '500' }}>Back</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.buttonContainer}>
            <View style={styles.buttons}>
              <TouchableOpacity
                style={styles.retakeButton}
                onPress={() => setShowCamera(true)}
              >
                <Text style={{ color: '#77c3ec', fontWeight: '500' }}>
                  Retake
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.usePhotoButton}
                onPress={extractFrames}
              >
                <Text style={{ color: 'white', fontWeight: '500' }}>
                  Extract Frames
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backButton: {
    backgroundColor: 'rgba(0,0,0,0.0)',
    position: 'absolute',
    justifyContent: 'center',
    width: '100%',
    top: 0,
    padding: 20,
  },
  backButtonStyle: {
    backgroundColor: 'rgba(0,0,0,0.2)',
    padding: 10,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#fff',
    width: 100,
  },
  buttonContainer: {
    backgroundColor: 'rgba(0,0,0,0.2)',
    position: 'absolute',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    bottom: 0,
    padding: 20,
  },
  buttons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
  },
  camButton: {
    height: 80,
    width: 80,
    borderRadius: 40,
    backgroundColor: '#B2BEB5',
    alignSelf: 'center',
    borderWidth: 4,
    borderColor: 'white',
  },
  retakeButton: {
    backgroundColor: '#fff',
    padding: 10,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#77c3ec',
  },
  usePhotoButton: {
    backgroundColor: '#77c3ec',
    padding: 10,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 10,
    borderWidth: 2,
    borderColor: 'white',
  },
  image: {
    width: '100%',
    height: '100%',
    aspectRatio: 9 / 16,
  },
});

export default App;