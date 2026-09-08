const version: string = process.version;
console.log(`Node version: ${version}`);

const arr: number[] = [1, 2, 3, 4, 5];
const mean: number = arr.reduce((a, b) => a + b, 0) / arr.length;

console.log(`Array: ${arr}`);
console.log(`Mean: ${mean}`);
console.log("✅ TypeScript is working");