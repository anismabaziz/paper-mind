import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
  {
    // A component that has to be scrolled to understand is a component with
    // more than one thing in it. components/ui/ is vendored component code kept
    // close to upstream, so it is not ours to reshape.
    files: ['src/features/**/*.{ts,tsx}', 'src/components/*.{ts,tsx}'],
    ignores: ['src/components/ui/**'],
    rules: {
      'max-lines': ['error', { max: 200 }],
    },
  },
)
