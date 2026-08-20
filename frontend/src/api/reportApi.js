import { mockReport } from "../mock/reportMock.js";
const mock = import.meta.env.VITE_USE_MOCK_API !== "false";
export async function generateReport(id) { if (mock) return mockReport(id); const response = await fetch(`/reports/${id}`, { method: "POST" }); if (!response.ok) throw new Error("Unable to generate the report."); return response.json(); }