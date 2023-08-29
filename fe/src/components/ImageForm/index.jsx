import React, { useState } from 'react';
import './ImageForm.css';
import axios from 'axios';
import CircularProgress from '@mui/material/CircularProgress';
import Switch from '@mui/material/Switch';

const ImageForm = () => {
  const [image, setImage] = useState(null);
  const [showDetectButton, setShowDetectButton] = useState(false);
  const [detectResult, setDetectResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [flag, setFlag] = useState(false);

  const handleImageChange = (event) => {
    const selectedImage = event.target.files[0];
    setImage(selectedImage);
    setShowDetectButton(true);
    setError(null);
  };

  const handleDetectClick = async () => {
    try {
      setError(null);
      setDetectResult(null);
      setLoading(true);

      const reader = new FileReader();
      reader.onload = async () => {
        const base64String = reader.result;
        try {
          let route = '';
          if (flag) {
            route = 'http://localhost:5000/ocr';
          } else {
            route = 'http://localhost:5000/ocr?single=True';
          }
          const response = await axios.post(route, { image: base64String }, {
            headers: {
              'Content-Type': 'application/json',
            },
          });

          setDetectResult(response.data.predictions);
          setError(null);
        } catch (error) {
          console.error('Error fetching data:', error);
          setError('An error occurred while fetching data. Please try again.');
        } finally {
          setLoading(false);
        }
      };
      reader.readAsDataURL(image);
    } catch (error) {
      console.error('Error reading image:', error);
    }
  };

  const handleSwitchChange = () => {
    setFlag(!flag);
  };

  return (
    <div className="image-form">
      <div className='introduction'>
        <span className='child-1'>HUỲNH TẤN LỢI - 51800574</span>
        <span className='child-2'>NGUYỄN HOÀNG QUANG NHẬT - 51800220</span>
        <span className='fancy'>NGHIÊN CỨU BÀI TOÁN OCR VÀ NHẬN DẠNG CHỮ VIẾT TAY ĐỊA CHỈ</span>
      </div>

      <div className="horizontal-line"></div>
      <div className="top-section">
        <div className="image-input">
          <label htmlFor="file-input" className="custom-file-input">
            Chọn hình ảnh
          </label>
          <input
            type="file"
            id="file-input"
            accept="image/*"
            onChange={handleImageChange}
          />
        </div>
        <div className="switch-container">
          <p>Ảnh một dòng/Ảnh nhiều dòng</p>
          <Switch checked={flag} onChange={handleSwitchChange} />
        </div>
      </div>
      <div className="bottom-section">
        {loading && <div className="loading-overlay"><CircularProgress /></div>}
        {image && (
          <div className='image-block'>
            <div className='block-1'>
              <img src={URL.createObjectURL(image)} alt="Uploaded" className="uploaded-image" />
            </div>
            <div className='block-2'>
              {showDetectButton && (
                <button className="detect-button" onClick={handleDetectClick}>
                  Nhận dạng chữ viết
                </button>
              )}
            </div>
            <div className='block-3'>
              {error && (
                <div className="error-message">
                  <p>{error}</p>
                </div>
              )}
              {detectResult && (
                <div className="detect-result">
                  <h2>Kết quả nhận dạng:</h2>
                  <ul>
                    {detectResult.map((result, index) => (
                      <li key={index}>{result}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageForm;
