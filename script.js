const API_URL =
  "https://keenkumquat-tweetsentimentclassifier.hf.space/predict";

document.getElementById("analyze").addEventListener("click", async () => {
    const tweet = document.getElementById("tweet").value;
    const result = document.getElementById("result");
    result.textContent = "Analyzing...";
  
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: tweet })
    });

  
    const data = await response.json();
    const { sentiment, confidence } = data;
  
    let color = sentiment === "Positive" ? "green" : sentiment === "Negative" ? "red" : "orange";
    result.style.color = color;
    result.textContent = `${sentiment} (${Math.round(confidence * 100)}% confident)`;
  });
  