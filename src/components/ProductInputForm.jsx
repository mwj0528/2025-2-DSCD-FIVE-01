import React, { useState } from 'react';
import { Box, TextField, Button, Typography } from '@mui/material';

function ProductInputForm({ onRecommend }) { 
  const [productName, setProductName] = useState('');
  const [productDesc, setProductDesc] = useState('');

  const handleSubmit = () => {
    // console.log 대신 부모로부터 받은 onRecommend 함수를 실행합니다.
    onRecommend();
  };

  return (
    <Box component="form" sx={{ '& > :not(style)': { m: 1, width: '100%' } }}>
      <Typography variant="h5" gutterBottom>
        상품 정보 입력
      </Typography>
      <TextField
        label="상품명"
        variant="outlined"
        value={productName}
        onChange={(e) => setProductName(e.target.value)}
      />
      <TextField
        label="상품에 대한 상세 설명 (재질, 용도 등)"
        variant="outlined"
        multiline
        rows={4}
        value={productDesc}
        onChange={(e) => setProductDesc(e.target.value)}
      />
      <Button variant="contained" size="large" onClick={handleSubmit}>
        추천받기
      </Button>
    </Box>
  );
}

export default ProductInputForm;