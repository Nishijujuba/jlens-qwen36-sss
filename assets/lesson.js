(() => {
  "use strict";

  function initializeQuiz(quiz) {
    const answer = quiz.dataset.answer;
    const explanation = quiz.dataset.explanation || "";
    const feedback = quiz.querySelector(".quiz-feedback");
    const buttons = [...quiz.querySelectorAll("button[data-choice]")];

    if (!answer || !feedback || buttons.length === 0) return;

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        buttons.forEach((item) => item.setAttribute("aria-pressed", "false"));
        button.setAttribute("aria-pressed", "true");

        const isCorrect = button.dataset.choice === answer;
        feedback.className = `quiz-feedback ${isCorrect ? "correct" : "incorrect"}`;
        feedback.textContent = isCorrect
          ? `正确。${explanation}`
          : `需要再想一步。${explanation}`;
      });
    });
  }

  document.querySelectorAll(".quiz[data-answer]").forEach(initializeQuiz);
})();
