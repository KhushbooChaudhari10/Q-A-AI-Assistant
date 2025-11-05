document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userMessageInput = document.getElementById("user-message");
    const sendButton = document.getElementById("send-button");

    const typingIndicator = document.getElementById("typing-indicator");

    sendButton.addEventListener("click", sendMessage);
    userMessageInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    function sendMessage() {
        const userMessage = userMessageInput.value.trim();
        if (userMessage === "") return;

        appendMessage(userMessage, "user");
        userMessageInput.value = "";
        typingIndicator.style.display = "flex";

        fetch("/get_response", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message: userMessage }),
        })
        .then(response => response.json())
        .then(data => {
            appendMessage(data.response, "bot");
            typingIndicator.style.display = "none";
        })
        .catch(error => {
            console.error("Error:", error);
            typingIndicator.style.display = "none";
        });
    }

    function appendMessage(message, sender) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        
        if (sender === "bot") {
            // Render markdown for bot messages
            const contentDiv = document.createElement("div");
            contentDiv.innerHTML = marked.parse(message);
            messageElement.appendChild(contentDiv);
        } else {
            // Plain text for user messages
            const p = document.createElement("p");
            p.textContent = message;
            messageElement.appendChild(p);
        }
        
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});