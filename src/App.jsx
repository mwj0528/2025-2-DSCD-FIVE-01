import React, { useState } from 'react';
import { Container, Box, Typography } from '@mui/material';
import ProductInputForm from './components/ProductInputForm';
import ResultList from './components/ResultList';

// 가짜 데이터(테스트용)
const DUMMY_RESULTS = [
  {
    hs_code: '8517.12',
    description: '스마트폰 (무선 통신 기능을 갖춘 단말기)',
    accuracy: '95.8%',
    reason: '관세율표 제8517호 해설서에 따라, 셀룰러 망이나 그 밖의 무선 망용 전화기는 이 호에 분류됩니다.'
  },
  {
    hs_code: '9102.11',
    description: '손목시계 (기계식 디스플레이만 가진 것)',
    accuracy: '88.2%',
    reason: '케이스가 귀금속으로 제작되지 않았으며, 기계식 디스플레이를 갖춘 시계는 제9102호에 해당합니다.'
  },
];

function App() {
  // 결과 데이터를 담을 state
  const [results, setResults] = useState([]);

  // 버튼이 클릭되면 실행될 함수
  const handleRecommend = () => {
    console.log("버튼 클릭! 가짜 데이터를 보여줍니다.");
    // results state를 가짜 데이터로 업데이트.
    setResults(DUMMY_RESULTS); 
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          HS Code 추천
        </Typography>
        
        {/* ProductInputForm에게 handleRecommend 함수를 'onRecommend'라는 이름으로 전달*/}
        <ProductInputForm onRecommend={handleRecommend} />
        
        {/* ResultList에게는 현재 results state를 전달 (처음엔 빈 배열) */}
        <ResultList results={results} />

      </Box>
    </Container>
  );
}

export default App;