import React from 'react';
import { Box, Typography, Card, CardContent, List, ListItem } from '@mui/material';

function ResultList({ results }) {
  if (!results || results.length === 0) {
    return null; // 결과가 없으면 아무것도 보여주지 않음
  }

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        추천 결과
      </Typography>
      <List>
        {results.map((item, index) => (
          <ListItem key={index} disablePadding>
            <Card variant="outlined" sx={{ width: '100%', mb: 2 }}>
              <CardContent>
                <Typography variant="h6" component="div">
                  HS Code: {item.hs_code} ({item.accuracy})
                </Typography>
                <Typography sx={{ mb: 1.5 }} color="text.secondary">
                  {item.description}
                </Typography>
                <Typography variant="body2">
                  <strong>분류 근거:</strong> {item.reason}
                </Typography>
              </CardContent>
            </Card>
          </ListItem>
        ))}
      </List>
    </Box>
  );
}

export default ResultList;